# ElderGuard — Humanoid Assistant Robot for Elderly Care

A fully **offline** autonomous assistant robot for elderly people. A **NAO V5** humanoid robot (the Body) is remotely controlled by a **Raspberry Pi 4** (the Brain), which runs all AI/ML inference on-device. The system recognizes voice commands from an authorized owner, tracks and follows them, detects falls, and can find and retrieve personal objects — all without internet connectivity.

> For a detailed technical deep-dive (design decisions, model justifications, protocol details, and examples), see [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [How It Works](#how-it-works)
- [Voice Commands](#voice-commands)
- [Hardware Setup (RPi + NAO)](#hardware-setup-rpi--nao)
- [PC Simulation Setup](#pc-simulation-setup)
- [Project Structure](#project-structure)
- [Models](#models)
- [Testing](#testing)
- [Configuration](#configuration)

---

## Architecture Overview

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
│  FSM Orchestrator (main.py)    │         │  motion_library.py               │
│  PID Visual Servo Controller   │         │    (wave, dance, pickup, offer)  │
└────────────────────────────────┘         └──────────────────────────────────┘
        ↕ Direct Ethernet (10.0.0.10)
```

**Key design principles:**
- **Brain/Body split** — RPi handles all AI; NAO executes physical actions. The only coupling is a JSON-over-TCP protocol.
- **Channel-based parallel execution** — LEGS, SPEECH, ARMS, and HEAD run on separate worker threads. The robot walks, talks, and waves simultaneously.
- **Dynamic model loading** — YOLOv8n is loaded only during object search and immediately unloaded, keeping steady-state RAM at ~470 MB on a 2 GB device.
- **Connection resilience** — heartbeat keepalive, watchdog safe-sit on connection loss, auto-reconnect with exponential backoff.

---

## How It Works

### Command Flow

```
User speaks ──→ Mic (44100 Hz) ──→ Resample (16 kHz) ──→ Vosk STT (grammar)
                                                              │
                                                    "hey nao" detected
                                                              │
                                                   NAO says "Yes?"
                                                              │
                                              Record command utterance
                                                              │
                                          ┌───────────────────┴───────────────────┐
                                          │                                       │
                                    Vosk recognizes                    ECAPA-TDNN computes
                                    command phrase                     speaker embedding
                                          │                                       │
                                          │                           Cosine similarity
                                          │                           vs enrolled master
                                          │                                       │
                                          └───────────────────┬───────────────────┘
                                                              │
                                                     Accepted (> 0.50)?
                                                        │          │
                                                       Yes         No → "I don't
                                                        │            recognize you"
                                                        │
                                              Parse intent ──→ Execute handler
                                                        │
                                              TCP command(s) to NAO
                                                        │
                                              NAO executes action
```

### FSM States

| State | What Happens |
|-------|-------------|
| **IDLE** | Vosk listens for wake word "hey nao". Fall monitor runs in background. |
| **LISTENING** | Captures command (5s timeout, 1.5s silence finalization). Records audio for verification. |
| **VERIFYING** | Computes speaker embedding, checks cosine similarity against enrolled owner. |
| **EXECUTING** | Parses intent, dispatches action(s) to NAO via TCP. |
| **SEARCHING** | Loads YOLOv8n, scans room by rotating head. Unloads YOLO on completion. Max 30s. |

**Fall detection** runs independently as an always-on daemon thread (5 Hz). It can interrupt any FSM state to alert about a detected fall.

---

## Voice Commands

| Command | Robot Action |
|---------|-------------|
| **"hey nao"** | Wake word — robot says "Yes?" and listens |
| **"follow me"** | Head-only tracking (watches person, safe default) |
| **"come here"** | Full follow — head tracking + body walking toward person |
| **"stop"** | Stops all activity |
| **"find my keys/phone/bottle/cup"** | Scans room with YOLO, reports result |
| **"bring me my keys/phone/bottle/cup"** | Finds, approaches, picks up, and delivers object |
| **"go to it" / "pick it up"** | Approaches last found object |
| **"wave hello"** | Wave animation |
| **"say hello"** | Greeting speech |
| **"sit down" / "stand up"** | Posture changes |
| **"what do you see"** | Describes visible objects |
| **"look left" / "look right"** | Turns head |
| **"turn around"** | 180-degree body rotation |
| **"introduce yourself"** | Animated self-introduction |
| **"dance"** | Dance animation |
| **"i'm okay"** | Acknowledges during fall alert |

---

## Hardware Setup (RPi + NAO)

### Prerequisites

| Device | Requirement |
|--------|------------|
| **NAO V5** | NAOqi SDK running on port 9559 |
| **Raspberry Pi 4** | 2 GB+ RAM, Python 3.9+, Raspbian OS |
| **USB Microphone** | Any USB mic (system will auto-detect sample rate) |
| **Pi Camera** | CSI or USB camera accessible via V4L2 (`/dev/video0`) |
| **Network** | Direct Ethernet cable between RPi and NAO |

### Step 1: Set Up the NAO Robot

1. **Copy server code to NAO:**
   ```bash
   scp -r nao_assistant/nao_body/ nao@10.0.0.10:~/assistant/
   scp start_server.sh nao@10.0.0.10:~/
   ```

2. **Start the server on NAO:**
   ```bash
   ssh nao@10.0.0.10
   cd ~/assistant && python server.py
   ```

   Or install auto-start (the script waits for NAOqi and network to be ready):
   ```bash
   # Add to NAO's autostart:
   chmod +x ~/start_server.sh
   # Configure NAOqi to run start_server.sh on boot
   ```

### Step 2: Set Up the Raspberry Pi

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd ElderGuard-Humanoid_Assistant_Robot/nao_assistant/rpi_brain
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --no-cache-dir -r requirements.txt
   ```

3. **Download and place models** in `rpi_brain/models/`:

   | Model | Source | Size | Destination |
   |-------|--------|------|-------------|
   | Vosk STT | [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) → `vosk-model-small-en-us-0.15` | ~65 MB | `models/vosk-model-small-en-us-0.15/` |
   | ECAPA-TDNN | Export using `tools/export_ecapa_onnx.py` (see below) | ~25 MB | `models/ecapa_tdnn.onnx` |
   | YOLOv8n INT8 | Export from ultralytics YOLOv8n → INT8 TFLite | 3.3 MB | `models/yolov8n_int8.tflite` |

   **Exporting the real ECAPA-TDNN model** (run on any PC with PyTorch):
   ```bash
   # On a PC (not on the RPi):
   pip install speechbrain torch onnx onnxruntime
   cd nao_assistant/tools
   python export_ecapa_onnx.py
   # Copy the output to RPi:
   scp ecapa_tdnn.onnx pi@<RPI_IP>:~/rpi_brain/models/
   ```

4. **Enroll your voice** (one-time setup):
   ```bash
   python enroll_speaker.py --person david --set-master
   # Speak 3 samples when prompted
   # Creates models/david_embedding.npy + models/master_embedding.npy
   ```

   Enroll additional speakers (optional, for multi-speaker mode):
   ```bash
   python enroll_speaker.py --person alice
   python enroll_speaker.py --list    # See all enrolled speakers
   ```

5. **Verify hardware** (optional but recommended):
   ```bash
   python tests/test_mic_fix.py       # Check mic sample rate
   python tests/test_models.py        # Verify all models load
   python tests/test_hardware.py      # Full mic + camera + TCP test
   ```

6. **Configure network** — edit `settings.py` or set environment variables:
   ```bash
   export NAO_IP=10.0.0.10      # NAO's IP address
   export NAO_PORT=5555          # TCP server port
   ```

### Step 3: Run the System

1. **On NAO** — ensure `server.py` is running (Step 1).

2. **On RPi:**
   ```bash
   cd nao_assistant/rpi_brain
   source venv/bin/activate
   python main.py
   ```

3. Say **"hey nao"** to activate, then speak a command.

---

## PC Simulation Setup

The simulation runs the **real brain code** on a PC with adapter classes that replace hardware-specific components. A mock NAO server runs on localhost, and an OpenCV dashboard provides real-time visualization.

### Prerequisites

- **Python 3.9+** (tested on 3.9–3.11)
- **Webcam** (optional — can run without camera)
- **Microphone** (optional — can use keyboard input instead)
- **Windows / Linux / macOS**

### Step 1: Create Virtual Environment

```bash
cd Simulation
python -m venv sim_venv

# Windows:
sim_venv\Scripts\activate

# Linux/macOS:
source sim_venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements_simulation.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `opencv-python` | Camera + GUI dashboard |
| `psutil` | RAM monitoring |
| `vosk` | Offline speech-to-text |
| `sounddevice` | Microphone capture |
| `onnxruntime` | Speaker verification |
| `mediapipe` (<0.10.20) | Face tracking + pose estimation |
| `ultralytics` | YOLOv8 object detection (replaces TFLite) |
| `pyttsx3` | PC text-to-speech (optional) |

### Step 3: Download Models

Place these in `nao_assistant/rpi_brain/models/`:

1. **Vosk model:**
   - Download `vosk-model-small-en-us-0.15` from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)
   - Extract to `nao_assistant/rpi_brain/models/vosk-model-small-en-us-0.15/`

2. **ECAPA-TDNN** (optional — simulation can skip verification):
   - The repo includes a 328 KB stub model for testing.
   - For real verification, run `tools/export_ecapa_onnx.py` on a PC.

3. **YOLOv8n:**
   - The simulation uses `ultralytics` which auto-downloads `yolov8n.pt` on first run.
   - Alternatively, place `yolov8n.pt` in the `Simulation/` directory.

### Step 4: Enroll a Speaker (Optional)

For real speaker verification in the simulation:

```bash
cd Simulation

# Option A: Enroll interactively (uses your microphone):
python enroll_sim.py --person david

# Option B: Enroll from pre-recorded WAV files:
python enroll_from_data.py --source augmented --verify
```

### Step 5: Run the Simulation

**Full mode** (mic + camera + speaker verification):
```bash
python run_simulation.py
```

**Keyboard-only mode** (no mic needed):
```bash
python run_simulation.py --keyboard
```

**No camera mode:**
```bash
python run_simulation.py --no-camera
```

**Skip speaker verification** (accept all commands):
```bash
python run_simulation.py --skip-verify
```

**Restrict to single master speaker:**
```bash
python run_simulation.py --master david
```

**Combine flags:**
```bash
python run_simulation.py --keyboard --no-camera --skip-verify
```

### Step 6: Using the Dashboard

The simulation opens an OpenCV window with:

- **Camera panel** — live webcam feed with face boxes, skeleton overlay, YOLO detections
- **State panel** — FSM state, NAO posture, channel states, head angles, servo mode
- **Demo console** — color-coded event log
- **Audio bar** — mic level + STT text + verify score
- **Hotkey bar** — keyboard shortcut reference

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `1`–`0` | Inject voice commands (follow me, stop, find keys, come here, sit, stand, what do you see, wave, dance, introduce) |
| `f` | Simulate robot fall (NAO tips over) |
| `p` | Simulate person fall (elderly user falls) |
| `r` | Recover from simulated falls |
| `d` | Simulate network disconnect |
| `q` | Quit |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SIM_CAMERA_INDEX` | `0` | Webcam device index |
| `SIM_SPEED_MULTIPLIER` | `1.0` | Speed up/slow down mock NAO motions |
| `SIM_PC_TTS` | `true` | Enable PC text-to-speech |
| `VOSK_MODEL_PATH` | Auto-detected | Override Vosk model path |

---

## Project Structure

```
ElderGuard-Humanoid_Assistant_Robot/
│
├── nao_assistant/
│   ├── nao_body/                      # NAO robot code (Python 2.7)
│   │   ├── server.py                  # TCP server + channel-based command dispatcher
│   │   ├── motion_library.py          # Motion presets (wave, dance, pickup, deliver)
│   │   └── tests/                     # Integration tests (phase 2, 4, 5)
│   │
│   ├── rpi_brain/                     # RPi brain code (Python 3.9+)
│   │   ├── main.py                    # FSM orchestrator (entry point)
│   │   ├── settings.py                # All tunable parameters
│   │   ├── enroll_speaker.py          # Voice enrollment utility
│   │   ├── requirements.txt           # RPi dependencies
│   │   ├── requirements_laptop.txt    # Laptop dependencies (no tflite-runtime)
│   │   ├── audio/                     # Mic capture, STT, speaker verification
│   │   ├── vision/                    # Camera, face tracking, YOLO, pose, fall detection
│   │   ├── servo/                     # PID visual servo controller
│   │   ├── command/                   # Intent parser
│   │   ├── comms/                     # TCP client + state cache
│   │   ├── utils/                     # RAM monitoring
│   │   ├── tests/                     # Integration + hardware tests
│   │   └── models/                    # Model binaries (git-ignored)
│   │
│   └── tools/
│       └── export_ecapa_onnx.py       # ECAPA-TDNN ONNX export script
│
├── Simulation/                        # PC simulation
│   ├── run_simulation.py              # Main entry point
│   ├── sim_config.py                  # PC settings overrides
│   ├── requirements_simulation.txt    # PC dependencies
│   ├── adapters/                      # PC hardware adapters (camera, YOLO, speaker verify)
│   ├── gui/                           # OpenCV dashboard (panels, event bus)
│   └── mock_nao/                      # Mock NAO server (uses real dispatcher)
│
├── start_server.sh                    # NAO auto-start script
├── PROJECT_SUMMARY.md                 # Detailed technical summary
├── MODELS_SETUP_GUIDE.md              # Model setup instructions
└── .gitignore
```

---

## Models

All models run on CPU. No GPU required.

| Model | Purpose | Size | Resident RAM | Loaded |
|-------|---------|------|-------------|--------|
| **Vosk** small-en-us-0.15 | Grammar-constrained speech-to-text | ~65 MB | ~100 MB | Always |
| **ECAPA-TDNN** (ONNX) | Speaker verification (192-dim embeddings) | ~25 MB | ~20 MB | Always |
| **MediaPipe BlazeFace** | Face detection for tracking | Built-in | ~40 MB | Always |
| **MediaPipe Pose** | Body keypoints for fall detection | Built-in | ~40 MB | Always |
| **YOLOv8n** INT8 (TFLite) | Object detection (COCO 80-class) | 3.3 MB | ~100 MB | Dynamic |

**Total steady-state:** ~470 MB. **Peak (with YOLO):** ~570 MB. Well within 2 GB RPi RAM.

---

## Testing

### Run Integration Tests

```bash
cd nao_assistant

# Server state machine and protocol tests
python -m pytest nao_body/tests/test_phase2.py -v      # 20 tests
python -m pytest nao_body/tests/test_phase4.py -v      # 20 tests
python -m pytest nao_body/tests/test_phase5.py -v      # 18 tests

# RPi brain tests
python -m pytest rpi_brain/tests/test_phase3.py -v     # 19 tests
python -m pytest rpi_brain/tests/test_phase6.py -v     # 16 tests
```

### Hardware Verification (on RPi)

```bash
cd nao_assistant/rpi_brain

python tests/test_mic_fix.py             # Discover mic sample rate
python tests/test_camera_find.py         # Find camera device
python tests/test_models.py              # Verify all models load
python tests/test_hardware.py            # Full hardware integration test
```

---

## Configuration

All parameters are in `rpi_brain/settings.py` and can be overridden via environment variables.

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `NAO_IP` | `10.0.0.10` | NAO robot IP |
| `NAO_PORT` | `5555` | TCP server port |
| `WAKE_WORD` | `"hey nao"` | Activation phrase |
| `SPEAKER_VERIFY_THRESHOLD` | `0.50` | Cosine similarity cutoff |
| `CAMERA_INDEX` | `0` | Camera device index |
| `FACE_LOST_FRAME_THRESHOLD` | `15` | Frames before "face lost" |
| `BODY_REALIGN_YAW_THRESHOLD_RAD` | `0.52` | Head angle before body turns (~30 deg) |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.40` | Min YOLO detection confidence |
| `YOLO_MAX_SEARCH_SECONDS` | `30.0` | Object search timeout |
| `FALL_DETECTION_ENABLED` | `true` | Toggle person fall detection |
| `FALL_CONFIRMATION_FRAMES` | `5` | Consecutive frames before fall alert |
| `HEARTBEAT_INTERVAL_S` | `2.0` | Keepalive interval |
| `RECONNECT_MAX_RETRIES` | `20` | Max reconnection attempts |

---

## License

This project was developed as part of the Humanoid Robot Development course at the university.
