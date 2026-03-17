# ElderGuard — Project Summary

## 1. Project Overview

**ElderGuard** is a fully offline autonomous assistant robot designed to help elderly people in their daily lives. The system uses a **NAO V5 humanoid robot** as the physical body, remotely controlled by a **Raspberry Pi 4** (2 GB RAM) that runs all AI and machine-learning inference. The two devices communicate over a direct Ethernet connection using a custom TCP protocol with newline-delimited JSON messages.

The robot can:
- Recognize and respond to voice commands from an authorized owner
- Track and follow the owner using real-time face detection
- Find and retrieve personal objects (keys, phone, bottle, cup)
- Detect if the elderly person has fallen and alert for help
- Perform social interactions (greet, introduce itself, wave, dance)

**All processing is done on-device with zero internet dependency**, making the system suitable for privacy-sensitive elderly care environments.

---

## 2. Architecture

### 2.1 Two-Device Split: Brain and Body

The system is split into two independent devices connected over Ethernet:

| Device | Role | Language | Responsibilities |
|--------|------|----------|-----------------|
| **Raspberry Pi 4** (2 GB) | Brain | Python 3.9+ | All AI inference (speech, vision, decision-making) |
| **NAO V5 Robot** | Body | Python 2.7 / NAOqi | Physical actions (walking, speaking, arm movements) |

**Why this split?**
- The NAO V5 runs Python 2.7 with NAOqi SDK — its onboard CPU cannot handle modern ML models.
- The RPi 4 runs Python 3.9+ with access to modern ML frameworks (MediaPipe, Vosk, ONNX Runtime, TFLite).
- Separating brain from body allows independent development, testing, and deployment.
- The only coupling between the two is the JSON-over-TCP protocol.

### 2.2 High-Level Architecture Diagram

```
RPi 4 "Brain" (Python 3.9+)               NAO V5 "Body" (Python 2.7 / NAOqi)
┌────────────────────────────────┐         ┌──────────────────────────────────┐
│                                │         │                                  │
│  USB Mic ──→ Vosk STT         │         │  TCP Server (port 5555)          │
│              (grammar mode)    │         │    ↓                             │
│         ──→ ECAPA-TDNN         │   TCP   │  CommandDispatcher               │
│              (speaker verify)  │ ──────► │    ├─ LEGS worker (walk, pose)   │
│                                │  JSON   │    ├─ SPEECH worker (say, TTS)   │
│  Pi Cam ──→ MediaPipe Face     │ ◄────── │    ├─ ARMS worker (wave, grip)   │
│              (face tracking)   │   ACK   │    └─ HEAD inline (move_head)    │
│         ──→ MediaPipe Pose     │         │                                  │
│              (fall detection)  │         │  Watchdog (safe-sit on timeout)  │
│         ──→ YOLOv8n            │         │  FallDetector (robot fall)       │
│              (object search)   │         │                                  │
│                                │         │  motion_library.py               │
│  FSM Orchestrator (main.py)    │         │    (wave, dance, pickup, offer)  │
│  PID Visual Servo Controller   │         │                                  │
└────────────────────────────────┘         └──────────────────────────────────┘
        ↕ Direct Ethernet (10.0.0.10)
```

### 2.3 Thread Architecture (RPi Brain)

The brain runs six concurrent threads:

```
Main Thread (FSM state machine)
  ├── Camera Thread          — continuous frame capture (always on)
  ├── Servo Thread           — 15 Hz PID face tracking (only during follow/come-here)
  ├── Fall Monitor Thread    — 5 Hz person fall detection (always on)
  ├── TCP Reader Thread      — async socket message dispatch
  └── Heartbeat Thread       — 2s keepalive to NAO watchdog
```

### 2.4 NAO Server: Channel-Based Command Routing

The NAO server routes commands to dedicated worker threads per body channel, enabling true parallel execution:

| Channel | Worker | Actions | Behavior |
|---------|--------|---------|----------|
| **LEGS** | Dedicated thread + queue | walk_toward, pose, rest, wake_up | New walk interrupts current walk |
| **SPEECH** | Dedicated thread + queue | say, animated_say | FIFO queue, non-blocking |
| **ARMS** | Dedicated thread + queue | animate (wave/dance), arm positions, hand open/close | Independent of legs |
| **HEAD** | Inline (no queue) | move_head, move_head_relative | Immediate execution, never blocked |
| **SYSTEM** | Inline | stop_all, stop_walk, query_state, heartbeat | Always available |

This means the robot can walk, talk, and wave simultaneously.

---

## 3. ML Models — What We Use and Why

All models run on the RPi 4 CPU. The 2 GB RAM constraint drove every model choice.

### 3.1 Vosk (Speech-to-Text)

| Property | Value |
|----------|-------|
| Model | `vosk-model-small-en-us-0.15` (~65 MB) |
| RAM | ~100 MB resident |
| Mode | Grammar-constrained (fixed phrase list) |

**Why Vosk?** It is one of the few STT engines that runs fully offline on ARM with low memory. Grammar-constrained mode restricts recognition to our known command phrases, achieving 90%+ accuracy even on a low-power device.

**Why grammar mode?** The robot only needs to understand ~20 specific phrases. Open-vocabulary STT would be slower, less accurate, and waste memory on capabilities we don't need.

### 3.2 ECAPA-TDNN (Speaker Verification)

| Property | Value |
|----------|-------|
| Model | ECAPA-TDNN via ONNX Runtime (~25 MB real / 328 KB stub) |
| RAM | ~20 MB resident |
| Output | 192-dimensional L2-normalized speaker embedding |
| Metric | Cosine similarity, threshold 0.50 |

**Why ECAPA-TDNN?** It is the state-of-the-art speaker verification model from the SpeechBrain ecosystem. It produces compact 192-dim embeddings that can distinguish speakers with a simple cosine similarity comparison.

**Why ONNX?** ONNX Runtime provides efficient CPU inference without requiring the full PyTorch stack (~2 GB), which would never fit on the RPi. The model is exported once on a PC using `tools/export_ecapa_onnx.py` and deployed as a lightweight `.onnx` file.

**Why a stub?** The repo ships with a 328 KB stub model for testing. The real model (~25 MB) must be exported on a PC and copied to the RPi. This keeps the repo lightweight and allows end-to-end pipeline testing without the real model.

### 3.3 MediaPipe BlazeFace (Face Detection)

| Property | Value |
|----------|-------|
| Model | MediaPipe BlazeFace (built-in) |
| RAM | ~40 MB resident |
| Speed | ~30 ms per frame on RPi |

**Why BlazeFace?** It is designed for mobile/edge devices, runs in real-time on ARM CPUs, and is included with MediaPipe (no separate model download needed). It provides the face bounding box that drives the PID visual servo controller.

### 3.4 MediaPipe Pose / BlazePose (Fall Detection)

| Property | Value |
|----------|-------|
| Model | MediaPipe Pose, complexity=0 (lightest) |
| RAM | ~40 MB resident |
| Output | 33 body keypoints (we use 9 key ones) |
| Speed | ~30-50 ms per frame on RPi |

**Why MediaPipe Pose?** It provides full-body keypoint detection at minimal computational cost. We use it specifically for person fall detection — monitoring body height ratio, drop velocity, aspect ratio, and torso angle.

**Why not always loaded?** The fall monitor runs its own instance. If RAM is tight, it falls back to face-bbox heuristic (no Pose model needed).

### 3.5 YOLOv8n INT8 (Object Detection)

| Property | Value |
|----------|-------|
| Model | YOLOv8n INT8 quantized TFLite (3.3 MB) |
| RAM | ~100 MB when loaded |
| Speed | ~6-7 FPS on RPi |
| Classes | COCO 80-class |
| Loaded | **Dynamically** — only during "find" commands |

**Why YOLOv8n?** It is the smallest YOLOv8 variant, and INT8 quantization reduces it to 3.3 MB while maintaining usable accuracy for common household objects.

**Why dynamic load/unload?** At ~100 MB resident, keeping YOLO loaded at all times would push total memory to ~570 MB, leaving insufficient headroom. By loading only during search and immediately unloading + GC, we keep steady-state at ~470 MB.

**Why TFLite?** TFLite Runtime is purpose-built for ARM inference. It's lighter than ONNX Runtime or full TensorFlow and optimized for the RPi's CPU.

### Memory Budget Summary

| Component | RAM | Resident? |
|-----------|-----|-----------|
| Python + OpenCV + OS | ~270 MB | Always |
| Vosk STT | ~100 MB | Always |
| ECAPA-TDNN | ~20 MB | Always |
| MediaPipe BlazeFace | ~40 MB | Always |
| MediaPipe Pose | ~40 MB | Always (fall monitor) |
| **Steady-state total** | **~470 MB** | |
| YOLOv8n (during search) | +100 MB | Dynamic |
| **Peak total** | **~570 MB** | |
| **Available on 2 GB RPi** | **~1.4 GB free** | |

---

## 4. Communication Protocol

### 4.1 Wire Format

```
{"action":"say","text":"Hello!"}\n
```

Newline-delimited JSON over TCP. Socket configured with `TCP_NODELAY=1`, 5s timeout, 4096-byte buffer.

### 4.2 Two Sending Modes

| Mode | Method | Behavior | Use Case |
|------|--------|----------|----------|
| **Blocking** | `send_command()` | Sends JSON, waits for response | Important commands (say, pose, walk) |
| **Fire-and-forget** | `send_fire_and_forget()` | Sends JSON, no wait | High-frequency commands (head moves at 15 Hz) |

### 4.3 Two-Phase ACK Protocol

For long-running commands (walk, pose, animate), the protocol uses two-phase acknowledgment:

```
RPi sends:  {"action":"walk_toward","x":1.0,"y":0,"theta":0,"id":"req-42"}
NAO returns: {"status":"ack","id":"req-42","state":{...}}      ← immediate
NAO returns: {"status":"done","id":"req-42","state":{...}}     ← when walk finishes
```

This allows the RPi to know the command was received (ack) and when it completed (done), without blocking the TCP connection.

### 4.4 Connection Resilience

- **Heartbeat**: RPi sends a `heartbeat` message every 2 seconds.
- **Watchdog**: If the NAO receives no messages for 10 seconds, it safely stops all motion, sits the robot down, and announces the connection loss.
- **Auto-reconnect**: On disconnect, the RPi attempts reconnection with exponential backoff (1s → 30s max, up to 20 attempts), then resyncs state via `query_state`.
- **Fall events**: The NAO server monitors `ALMemory` for physical robot falls and sends async `{"type":"event","event":"fallen"}` to the RPi.

---

## 5. Finite State Machine (FSM)

The brain operates as a state machine with 5 active states:

```
                              ┌─────────────────────────────────────────────────────────┐
                              │            FALL DETECTED (any state)                     │
                              │  → Stop all → "Are you okay?" → Listen 10s              │
                              │  → "I'm okay" heard → Resume                            │
                              │  → No response → "Calling for help!"                    │
                              └─────────────────────────────────────────────────────────┘

  ┌──────┐   "hey nao"   ┌───────────┐   speech   ┌───────────┐  owner OK  ┌───────────┐
  │ IDLE │ ────────────→  │ LISTENING  │ ────────→  │ VERIFYING  │ ────────→  │ EXECUTING │
  └──┬───┘                └─────┬─────┘            └─────┬─────┘            └─────┬─────┘
     │                     timeout│                 rejected│                      │
     │                          │                       │                    "find my X"
     │                          │                       │                         │
     │                          │                       │                  ┌──────┴──────┐
     │                          │                       │                  │  SEARCHING   │
     │                          │                       │                  └──────┬──────┘
     │                          │                       │                    found│timeout
     └──────────────────────────┴───────────────────────┴─────────────────────────┘
                                          back to IDLE
```

| State | Duration | What Happens |
|-------|----------|-------------|
| **IDLE** | Indefinite | Vosk listens for wake word "hey nao". Fall monitor runs in background. |
| **LISTENING** | 5s max | Captures command utterance. 1.5s silence ends recording early. Also records raw audio for speaker verification. |
| **VERIFYING** | <1s | ECAPA-TDNN computes speaker embedding, compares to master. Accept (>0.50) or reject. |
| **EXECUTING** | Varies | Parses intent, dispatches handler. Sends TCP commands to NAO. |
| **SEARCHING** | 30s max | Loads YOLOv8n, scans by rotating head through 5 angles. Unloads YOLO on completion. |

**Fall detection** is independent of the FSM — it runs as an always-on daemon thread and can interrupt ANY state.

---

## 6. Key Design Decisions

### 6.1 Grammar-Constrained STT Over Open Vocabulary

We chose to restrict Vosk to a fixed grammar of ~25 phrases rather than open-vocabulary recognition. This means the system can only understand pre-defined commands, but with much higher accuracy (90%+) on a resource-constrained device. For an elderly care robot with a fixed command set, this trade-off strongly favors reliability.

### 6.2 Dynamic YOLO Loading Over Persistent Loading

Object detection is only needed during "find" commands (~1% of operating time). Loading YOLOv8n permanently would waste 100 MB of RAM. We dynamically load the model on demand and immediately unload it after search completion, keeping steady-state memory 100 MB lower.

### 6.3 PID Visual Servo Over Fixed Movement

Rather than simple "turn left/right" commands, we implemented a continuous PID control loop for face tracking. Two PID controllers (yaw and pitch) compute angular corrections at ~15 Hz, producing smooth, natural head movements that keep the person centered in frame. This is critical for a natural-feeling interaction.

### 6.4 Channel-Based Parallel Execution Over Serial Dispatch

Early versions of the NAO server processed commands serially, meaning "say hello" would block walking. We redesigned the server with dedicated worker threads per body channel (LEGS, SPEECH, ARMS, HEAD). This allows the robot to walk, talk, and gesture simultaneously, creating much more natural behavior.

### 6.5 Two-Phase ACK Over Simple Request-Response

Long-running commands (walking 2 meters, sitting down) can take several seconds. A simple blocking protocol would freeze the RPi during this time. The two-phase ACK (immediate acknowledgment + completion notification) allows the RPi to pipeline commands and react to events during long operations.

### 6.6 Person Fall Detection via Temporal Fusion

A single frame of "person looks low" could be a false positive (bending over, sitting). We use a 4-signal temporal fusion approach:

1. **Height ratio** — current height / baseline (40% weight)
2. **Drop velocity** — rate of height decrease (30% weight)
3. **Aspect ratio** — body width / height (15% weight)
4. **Torso angle** — shoulder-to-hip angle from vertical (15% weight)

This combination, requiring 5 consecutive high-score frames, distinguishes real falls (<1 second) from voluntary sit-downs (2-3 seconds).

### 6.7 Head-Only Tracking vs. Full Follow

We split "follow me" (head-only tracking, safe default) from "come here" (head + body walking). For elderly users, the robot watching them without moving is the safer default. Walking toward the person requires an explicit "come here" command.

### 6.8 Watchdog Safe-Sit on Connection Loss

If the RPi crashes or disconnects, the robot could be stuck mid-walk. The NAO watchdog monitors heartbeats and automatically sits the robot down after 10 seconds of silence, preventing potential safety hazards.

---

## 7. Voice Commands

| Voice Phrase | What the Robot Does |
|-------------|-------------------|
| "hey nao" | Wakes up, says "Yes?" and listens for a command |
| "follow me" | Starts head-only tracking (watches person, does not walk) |
| "come here" | Walks toward person with full head+body tracking |
| "stop" | Stops all activity (walking, speaking, tracking) |
| "find my keys/phone/bottle/cup" | Loads YOLO, scans room, reports result |
| "bring me my keys/phone/bottle/cup" | Finds, approaches, picks up, delivers object |
| "go to it" / "pick it up" | Approaches and picks up last found object |
| "wave hello" | Plays wave animation |
| "say hello" | Says "Hello! Nice to see you." |
| "sit down" | Sits the robot down |
| "stand up" | Stands the robot up |
| "what do you see" | Loads YOLO, describes visible objects |
| "look left" / "look right" | Turns head, then re-centers |
| "turn around" | 180-degree body turn |
| "introduce yourself" | Animated self-introduction speech |
| "dance" | Plays dance animation |
| "i'm okay" | Acknowledges during fall response |

---

## 8. Example Scenarios

### Example 1: "Find my keys"

```
1. User says "hey nao"
2. Vosk detects wake word → FSM transitions to LISTENING
3. NAO says "Yes?"
4. User says "find my keys"
5. Vosk recognizes "find my keys" → FSM transitions to VERIFYING
6. ECAPA-TDNN computes embedding → cosine sim = 0.72 > 0.50 → accepted
7. Parser maps "find my keys" → IntentType.FIND_OBJECT (target: ["cell phone", "remote"])
8. FSM transitions to SEARCHING
9. YOLOv8n loaded (~100 MB), head scans through 5 angles
10. At angle 0.5 rad: YOLO detects "remote" at confidence 0.65
11. NAO says "I found your keys! They are to my right."
12. YOLOv8n unloaded, GC runs → RAM drops back to ~470 MB
13. FSM transitions to IDLE
```

### Example 2: "Come here" with person lost

```
1. User says "hey nao" → "come here" → verified
2. Robot starts full-follow mode: PID head tracking + body walking
3. set_walk_velocity commands sent every 400ms (steering based on head angle)
4. User walks behind a corner → face lost for 15 frames
5. Robot stops walking, starts person search:
   a. Head scan: 5 angles over ~5 seconds → face not found
   b. Body rotation: 360° turn over ~22 seconds
   c. At 180°: face re-detected
6. Robot resumes walking toward person
7. User says "hey nao" → "stop"
8. Robot stops all motion, returns to IDLE
```

### Example 3: Person falls

```
1. Robot is in IDLE state (or any state)
2. Fall monitor detects: height ratio drops to 0.35, velocity = 0.45/s
3. Fall score exceeds threshold for 5 consecutive frames → TRIGGERED
4. Fall event interrupts current FSM state
5. Robot stops all activity (stop_all)
6. NAO says "Are you okay? I think you may have fallen!"
7. Robot listens for 10 seconds:
   a. User says "I'm okay" → Robot says "Good to hear!" → resume
   b. No response → Robot says "I'm calling for help!" → escalation
8. If person stands back up (height > 80% baseline for 2s) → fall detector re-arms
```

### Example 4: "Bring me my phone"

```
1. User says "hey nao" → "bring me my phone" → verified
2. Robot checks last found object — phone was found 30 seconds ago
3. Phase 1: Approach — walks toward stored object angle
4. Phase 2: Pickup — crouch → reach down → close hand → stand
5. Phase 3: Find person — starts full-follow servo to locate owner
6. Phase 4: Deliver — extends arm → waits → opens hand → rests arm
7. Robot says "Here you go!" → returns to IDLE
```

---

## 9. PC Simulation

A complete PC-based simulation allows running and testing the entire system on a single computer without physical hardware. See [Simulation Setup](#simulation-setup-pc) in the README for instructions.

### 9.1 How the Simulation Works

The simulation runs the **real brain code** (`main.py`, all modules) with PC-compatible adapter classes swapped in at import time:

| Real Component | PC Adapter | How |
|---------------|------------|-----|
| Pi Camera (V4L2) | PC Webcam (OpenCV) | `PcCamera` — 640x480 @ 30 FPS |
| TFLite YOLOv8n | Ultralytics YOLOv8n | `PcObjectDetector` — .pt model |
| ONNX Speaker Verify | SpeechBrain (PyTorch) | `PcSpeakerVerifier` — native embeddings |
| NAO Robot (NAOqi) | Mock NAO Server | `MockMotion`, `MockTts` — simulated physics |
| USB Mic + Vosk | Same (or keyboard fallback) | Mic works natively on PC |

The key module is `adapters/bootstrap.py`, which:
1. Installs `sim_config.py` as the `settings` module (overriding RPi-specific values)
2. Adds RPi brain paths to `sys.path`
3. Stubs the `naoqi` module (NAO SDK not available on PC)
4. Monkey-patches hardware classes with PC adapters

### 9.2 Mock NAO Server

The mock server uses the **real** `NaoStateMachine`, `CommandDispatcher`, and TCP server code from `nao_body/server.py`. Only the NAOqi proxy classes are mocked:

- **MockMotion** — tracks joint angles and walk state, logs all calls
- **MockTts** — optionally speaks via `pyttsx3` on PC
- **MockMemory** — simulates ALMemory (fall state, posture)

### 9.3 GUI Dashboard

An OpenCV-based dashboard provides real-time visualization:

- **Camera panel** — live webcam feed with face detection overlay, skeleton overlay (fall detection), YOLO bounding boxes
- **State panel** — FSM state, NAO posture, channel states, head angles, servo mode, fall monitor status
- **Demo console** — color-coded event log showing STT results, commands sent, NAO responses
- **Audio bar** — mic level, STT partial text, speaker verification score
- **Hotkey bar** — keyboard shortcuts for injecting commands

### 9.4 Hotkeys

| Key | Action |
|-----|--------|
| `1` | follow me |
| `2` | stop |
| `3` | find my keys |
| `4` | come here |
| `5` | sit down |
| `6` | stand up |
| `7` | what do you see |
| `8` | wave hello |
| `9` | dance |
| `0` | introduce yourself |
| `f` | Inject robot fall event |
| `p` | Inject person fall (synthetic pose) |
| `r` | Recover from falls |
| `d` | Simulate disconnect |
| `q` | Quit |

---

## 10. Testing

### Integration Test Suites

| Suite | Tests | Covers |
|-------|-------|--------|
| `test_phase2.py` | 20 | Server state machine, ACK protocol |
| `test_phase3.py` | 19 | RPI state cache, guards, async reader |
| `test_phase4.py` | 20 | Channel routing, walk interruption, stop_all |
| `test_phase5.py` | 18 | Heartbeat, watchdog, reconnect, fall events |
| `test_phase6.py` | 16 | Speaker verification, embeddings, multi-speaker |
| **Total** | **93** | |

### Hardware Test Scripts

| Script | Purpose |
|--------|---------|
| `test_hardware.py` | Full mic + camera + TCP integration |
| `test_mic_fix.py` | Discover supported mic sample rates |
| `test_camera_find.py` | Scan /dev/video* for Pi camera |
| `test_models.py` | Verify all 4 models load and produce output |

---

## 11. Project Structure

```
ElderGuard-Humanoid_Assistant_Robot/
├── nao_assistant/
│   ├── nao_body/                      # Deployed to NAO (Python 2.7)
│   │   ├── server.py                  # TCP server + channel-based command dispatcher
│   │   ├── motion_library.py          # Motion presets (wave, dance, pickup, deliver)
│   │   └── tests/                     # Integration tests (phase 2, 4, 5)
│   │
│   ├── rpi_brain/                     # Deployed to RPi (Python 3.9+)
│   │   ├── main.py                    # FSM orchestrator (entry point)
│   │   ├── settings.py                # All tunable parameters
│   │   ├── enroll_speaker.py          # Voice enrollment utility
│   │   ├── requirements.txt           # RPi pip dependencies
│   │   ├── requirements_laptop.txt    # Laptop pip dependencies (no tflite-runtime)
│   │   │
│   │   ├── audio/
│   │   │   ├── mic_stream.py          # Ring-buffer mic capture (44100→16000 Hz resample)
│   │   │   ├── stt_engine.py          # Vosk grammar-constrained STT
│   │   │   └── speaker_verify.py      # ECAPA-TDNN speaker verification (ONNX)
│   │   │
│   │   ├── vision/
│   │   │   ├── camera.py              # Background V4L2 frame grabber
│   │   │   ├── face_tracker.py        # MediaPipe BlazeFace
│   │   │   ├── object_detector.py     # YOLOv8n TFLite (dynamic load/unload)
│   │   │   ├── pose_estimator.py      # MediaPipe Pose wrapper
│   │   │   ├── fall_detector.py       # 4-signal fall detection logic
│   │   │   └── fall_monitor.py        # Always-on fall monitor daemon
│   │   │
│   │   ├── servo/
│   │   │   └── visual_servo.py        # PID head tracking + body realignment
│   │   │
│   │   ├── command/
│   │   │   └── parser.py              # Grammar phrase → IntentType mapping
│   │   │
│   │   ├── comms/
│   │   │   └── tcp_client.py          # Thread-safe TCP client + state cache + auto-reconnect
│   │   │
│   │   ├── utils/
│   │   │   └── memory.py              # RAM monitoring, GC helpers
│   │   │
│   │   ├── tests/                     # Integration + hardware tests
│   │   │
│   │   └── models/                    # Git-ignored model binaries
│   │       ├── vosk-model-small-en-us-0.15/
│   │       ├── ecapa_tdnn.onnx
│   │       ├── yolov8n_int8.tflite
│   │       └── *_embedding.npy        # Enrolled speaker embeddings
│   │
│   └── tools/
│       └── export_ecapa_onnx.py       # Export real ECAPA-TDNN to ONNX (run on PC)
│
├── Simulation/                        # PC-based simulation
│   ├── run_simulation.py              # Main entry point
│   ├── sim_config.py                  # PC settings overrides
│   ├── requirements_simulation.txt    # PC pip dependencies
│   ├── enroll_from_data.py            # Batch enrollment from WAV files
│   ├── enroll_sim.py                  # Interactive enrollment
│   ├── export_ecapa_model.py          # ECAPA model export
│   │
│   ├── adapters/                      # PC hardware adapters
│   │   ├── bootstrap.py               # Import patcher (the key module)
│   │   ├── pc_camera.py               # Webcam adapter
│   │   ├── pc_object_detector.py      # Ultralytics YOLO adapter
│   │   ├── pc_pose_estimator.py       # Pose + synthetic fall injection
│   │   ├── pc_speaker_verify.py       # SpeechBrain verifier
│   │   └── keyboard_input.py          # Keyboard command fallback
│   │
│   ├── gui/                           # OpenCV dashboard
│   │   ├── dashboard.py               # Main window compositor
│   │   ├── panels.py                  # Panel renderers
│   │   ├── event_bus.py               # Thread-safe event stream
│   │   └── robot_visualizer.py        # Robot stick-figure visualization
│   │
│   └── mock_nao/                      # Mock NAO server
│       ├── mock_server.py             # Server bootstrap
│       └── mock_proxies.py            # Mock ALProxy classes
│
├── start_server.sh                    # NAO auto-start script
├── .gitignore
└── README.md
```
