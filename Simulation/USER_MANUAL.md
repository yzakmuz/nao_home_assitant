# ElderGuard Simulation — User Manual

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Installation](#2-installation)
3. [First-Time Setup](#3-first-time-setup)
4. [Speaker Enrollment](#4-speaker-enrollment)
5. [Running the Simulation](#5-running-the-simulation)
6. [Dashboard Overview](#6-dashboard-overview)
7. [Using Voice Commands](#7-using-voice-commands)
8. [Using Keyboard Commands](#8-using-keyboard-commands)
9. [Hotkey Reference](#9-hotkey-reference)
10. [Demo Scenarios](#10-demo-scenarios)
11. [Configuration](#11-configuration)
12. [Troubleshooting](#12-troubleshooting)
13. [How It Works](#13-how-it-works)

---

## 1. System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10/11, Linux, macOS | Windows 10/11 |
| CPU | Any x64 processor | Intel i5 / AMD Ryzen 5 or better |
| RAM | 4 GB | 8 GB |
| Webcam | Optional (synthetic mode available) | Any USB webcam or built-in |
| Microphone | Optional (keyboard mode available) | Any USB microphone |
| Speakers | Optional | For hearing NAO's speech responses |
| Disk | ~500 MB (models + dependencies) | ~1 GB |
| Internet | Required for first-time setup only | Not needed after setup |

### Software

- Python 3.9 or later (3.10+ recommended)
- pip (Python package manager)
- Git (for cloning the repository)

---

## 2. Installation

### Step 1: Navigate to the Simulation Directory

```bash
cd path/to/ElderGuard-Humanoid_Assistant_Robot/Simulation
```

### Step 2: Create a Virtual Environment

> **Windows Long-Path Warning:** MediaPipe and other packages fail when
> installed in deeply nested directories. Always create the venv in a
> **short path** (e.g., `C:\Users\<you>\sim_venv`), NOT inside this
> project folder.

```bash
# Create venv in a SHORT path (replace <you> with your username):
python -m venv C:\Users\<you>\sim_venv

# Activate — pick the line that matches your shell:

# PowerShell:
C:\Users\<you>\sim_venv\Scripts\Activate.ps1

# CMD:
C:\Users\<you>\sim_venv\Scripts\activate.bat

# Git Bash:
source C:/Users/<you>/sim_venv/Scripts/activate

# Linux/macOS (no long-path issue — can use local venv):
python -m venv venv && source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Make sure you are in the Simulation directory and the venv is active:
pip install -r requirements_simulation.txt
```

This installs:
- `vosk` — Offline speech-to-text engine
- `sounddevice` — Microphone access
- `numpy` — Array operations
- `onnxruntime` — Speaker verification inference
- `mediapipe` — Face detection
- `opencv-python` — Camera + GUI dashboard
- `psutil` — Memory monitoring
- `ultralytics` — YOLOv8 object detection
- `pyttsx3` — PC text-to-speech (optional but recommended)

---

## 3. First-Time Setup

### Download the Vosk Speech Model

The simulation needs the Vosk English model for voice recognition.

1. Download `vosk-model-small-en-us-0.15` from:
   https://alphacephei.com/vosk/models

2. Extract the folder to:
   ```
   nao_assistant/rpi_brain/models/vosk-model-small-en-us-0.15/
   ```

3. Verify the path contains files like `am/final.mdl`, `conf/mfcc.conf`, etc.

**If you skip this step:** The simulation will automatically fall back to
keyboard input mode (you type commands instead of speaking them).

### Speaker Verification Model (Optional)

For real speaker verification, you need the ECAPA-TDNN model:

1. On a PC with PyTorch installed, run:
   ```bash
   python nao_assistant/tools/export_ecapa_onnx.py
   ```

2. Copy the output `ecapa_tdnn.onnx` (~25 MB) to:
   ```
   nao_assistant/rpi_brain/models/ecapa_tdnn.onnx
   ```

3. Run enrollment:
   ```bash
   python nao_assistant/rpi_brain/enroll_speaker.py
   ```

**If you skip this step:** The simulation will skip speaker verification
(all commands are accepted regardless of who speaks).

### YOLOv8 Model (Auto-Download)

The `ultralytics` package automatically downloads the YOLOv8n model
(~6 MB) on first use.  This requires a one-time internet connection.

To pre-download:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## 4. Speaker Enrollment

Speaker enrollment teaches the system to recognize authorized voices. The system
supports **multiple enrolled speakers** — each person gets their own voice profile.

### Option A: Enroll from Pre-Recorded Data (Recommended)

If you have pre-recorded voice data in the `data/` folder:

```bash
# Enroll all speakers from augmented data (best quality — uses 252 files total):
python enroll_from_data.py --source augmented --verify

# Enroll from processed data (fewer files, faster):
python enroll_from_data.py --source processed --verify

# Enroll only one person:
python enroll_from_data.py --source augmented --person david

# List who is enrolled:
python enroll_from_data.py --list

# Set a person as the master speaker:
python enroll_from_data.py --set-master david
```

### Option B: Enroll Live from Microphone

Record voice samples interactively (3 samples per person, 4 seconds each):

```bash
# Enroll one person:
python enroll_sim.py --person david --set-master

# Enroll both David and Itzhak:
python enroll_sim.py --enroll-both --set-master david

# Custom settings (5 samples, 5 seconds each):
python enroll_sim.py --person david --samples 5 --duration 5

# Test your voice against all enrolled speakers:
python enroll_sim.py --test-only

# List enrolled speakers:
python enroll_sim.py --list
```

### Multi-Speaker Mode vs Single Master

The system supports two verification modes:

| Mode | How to activate | Behavior |
|------|----------------|----------|
| **Multi-speaker** (default) | `python run_simulation.py` | Accepts commands from ALL enrolled speakers. Identifies who spoke. |
| **Single master** | `python run_simulation.py --master david` | Only accepts commands from the specified person. |

**Key points:**
- The `--master` flag is **per-session** — it does NOT permanently change who is enrolled
- When you omit `--master`, the system accepts ALL enrolled speakers again
- You can switch masters between sessions freely:
  ```bash
  python run_simulation.py --master david    # only david
  python run_simulation.py --master itzhak   # only itzhak
  python run_simulation.py                   # both accepted
  ```
- Unknown voices (people who are not enrolled) are always rejected

### Adding a New Person

To add a new person to the system:

1. **From data files:** Create a folder `data/augmented/<name>/` with WAV recordings,
   then run `python enroll_from_data.py --person <name>`
2. **From microphone:** Run `python enroll_sim.py --person <name>`
3. The new person is immediately available in multi-speaker mode

### Transferring Embeddings to RPi (Real Hardware)

Embeddings generated on PC are directly portable to the real RPi:
```bash
# Copy all speaker embeddings to the RPi:
scp nao_assistant/rpi_brain/models/*_embedding.npy pi@<RPI_IP>:~/rpi_brain/models/
```

The same multi-speaker verification works on both PC and RPi.

---

## 5. Running the Simulation

### Basic Start (Full Mode)

```bash
python run_simulation.py
```

This starts with:
- Real microphone for voice commands
- Real webcam for face tracking
- Speaker verification enabled (if model exists)
- PC text-to-speech enabled
- Dashboard at 1440x900

### Keyboard-Only Mode (No Mic/Camera Needed)

```bash
python run_simulation.py --keyboard --no-camera
```

Best for:
- Quick testing without hardware
- Noisy environments
- PCs without webcam/mic

### Common Launch Configurations

| Scenario | Command |
|----------|---------|
| Full demo (voice + camera) | `python run_simulation.py` |
| Single master (david only) | `python run_simulation.py --master david` |
| Multi-speaker (all enrolled) | `python run_simulation.py` |
| Keyboard + camera | `python run_simulation.py --no-mic --no-verify` |
| Keyboard only | `python run_simulation.py --keyboard --no-camera` |
| Fast demo (2x speed) | `python run_simulation.py --speed 0.5` |
| Slow-motion analysis | `python run_simulation.py --speed 2.0` |
| External camera | `python run_simulation.py --camera 1` |
| Silent (no TTS) | `python run_simulation.py --no-tts` |
| Debug logging | `python run_simulation.py --verbose` |

### All CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--no-mic` | Off | Disable microphone, use keyboard input |
| `--no-camera` | Off | Use synthetic camera (moving dot) |
| `--no-verify` | Off | Skip speaker verification |
| `--no-tts` | Off | Disable PC text-to-speech |
| `--no-yolo` | Off | Disable object detection |
| `--keyboard` | Off | Shortcut for `--no-mic --no-verify` |
| `--camera N` | 0 | Camera device index |
| `--speed X` | 1.0 | Speed multiplier (lower = faster) |
| `--master NAME` | (none) | Restrict to single speaker (e.g., `--master david`). Omit for multi-speaker mode. |
| `--verbose` | Off | Enable debug-level logging |

### Stopping the Simulation

- Press `q` in the dashboard window, OR
- Press `Ctrl+C` in the terminal, OR
- Close the dashboard window

The simulation will gracefully shut down all threads (brain, server, workers).

---

## 5. Dashboard Overview

The dashboard is an OpenCV window (1440x900) with six panels:

### Camera Panel (Top-Left, 640x480)

Shows the live camera feed with overlays:
- **Green rectangle** — Detected face (MediaPipe BlazeFace)
- **Green skeleton** — Body keypoints and connections (MediaPipe Pose, when person detected)
- **Red skeleton + "FALL DETECTED" banner** — When person fall is detected (flashing)
- **Blue rectangles** — YOLO object detections (during "find" or "what do you see")
- **Red crosshair** — Image center (servo target)
- **Yellow line** — Vector from face center to image center (PID error)

If using synthetic camera mode (`--no-camera`), shows a dark background
with a moving colored dot.

### State Panel (Top-Right)

Real-time NAO state information:
- **FSM State** — Current state machine state: IDLE, LISTENING, VERIFYING, EXECUTING, SEARCHING
- **NAO Posture** — standing, sitting, resting, fallen
- **Channel states** — HEAD, LEGS, SPEECH, ARMS (each shows idle/active)
- **Head angles** — Current yaw and pitch in radians
- **PID error** — Visual servo error (distance from face to center)
- **Servo status** — ON/OFF
- **Fall Monitor** — MONITORING (green), CALIBRATING (yellow), TRIGGERED (red flashing), RECOVERY (yellow), INACTIVE (gray)
- **Fall Score** — 0.00-1.00 fusion score (shown when monitoring)

Color coding:
- Green = idle/ok
- Yellow = active/busy
- Red = error/fallen

### Robot Visualization (Bottom-Left)

A 2D stick figure representing the NAO robot:
- Head rotates/tilts based on actual head angles
- Body changes height for sitting vs standing
- Arms animate during wave/dance
- Arrow indicates walk direction when legs are active

### Command Log (Bottom-Right)

Scrolling list of the last 20 commands sent to NAO:
- Timestamp
- Action name and key parameters
- Status (OK / ERROR / REJECTED)

### Audio Bar (Bottom Strip)

- **MIC level** — Real-time audio amplitude meter
- **STT text** — Last recognized speech text
- **VERIFY score** — Last speaker verification score with ACCEPTED/REJECTED

### Hotkey Bar (Very Bottom)

Quick reference of available keyboard shortcuts.

---

## 6. Using Voice Commands

### Voice Pipeline Flow

```
1. Say "hey nao"          → NAO responds "Yes?"
2. Say your command        → Vosk recognizes it
3. Speaker verification    → ECAPA-TDNN checks your voice
4. Command executes        → NAO performs the action
```

### Supported Voice Commands

| Say This | What Happens |
|----------|-------------|
| "follow me" | NAO starts head-only tracking (watches you, no walking) |
| "stop" | NAO stops all movement and tracking |
| "find my keys" | NAO searches for keys by scanning with YOLO |
| "find my phone" | NAO searches for a phone |
| "find my bottle" | NAO searches for a bottle |
| "find my cup" | NAO searches for a cup |
| "wave hello" | NAO performs a wave animation |
| "say hello" | NAO says "Hello! Nice to see you." |
| "sit down" | NAO sits down (stops tracking first) |
| "stand up" | NAO stands up |
| "what do you see" | NAO describes visible objects |
| "look left" | NAO looks left, then re-centers |
| "look right" | NAO looks right, then re-centers |
| "turn around" | NAO rotates 180 degrees |
| "come here" | NAO starts full follow mode (head + body walking, person search when face lost) |
| "introduce yourself" | NAO gives a long animated introduction |
| "dance" | NAO performs a dance sequence |
| "bring me my phone/keys/bottle/cup" | Find object, pick it up, bring it to you (full 5-phase sequence) |
| "go to it" | Approach and pick up the last found object |
| "pick it up" | Same as "go to it" — approach and pick up last found object |
| "i'm okay" | Acknowledges you're fine (used during fall response, or anytime) |

### Tips for Voice Recognition

- Speak clearly and at normal volume
- Wait for "Yes?" before saying your command
- Avoid background noise
- The system uses grammar-constrained recognition — only the commands listed
  above are recognized (this is intentional for accuracy)
- If recognition fails, NAO will say "I did not catch that" — just try again

---

## 7. Using Keyboard Commands

When running with `--keyboard` or `--no-mic`, commands are typed in the
terminal instead of spoken.

### How It Works

1. The terminal shows: `[SIM] Press Enter to simulate wake word 'hey nao'...`
2. Press Enter
3. The terminal shows: `[SIM] Type command: `
4. Type a command (e.g., `follow me`) and press Enter
5. The command is processed through the same pipeline as voice

### Example Session

```
[SIM] Press Enter to simulate wake word 'hey nao'...
[Enter]
[SIM] Type command: follow me
[INFO] Command recognized: 'follow me'
[INFO] Executing intent: FOLLOW_ME

[SIM] Press Enter to simulate wake word 'hey nao'...
[Enter]
[SIM] Type command: find my phone
[INFO] Command recognized: 'find my phone'
[INFO] Executing intent: FIND_OBJECT
```

---

## 8. Hotkey Reference

When the dashboard window is focused, these keys are available:

### Command Injection (Bypasses Voice Pipeline)

| Key | Command | Description |
|-----|---------|-------------|
| `1` | follow me | Start face tracking and following |
| `2` | stop | Stop all movement and tracking |
| `3` | find my phone | Search for phone with YOLO |
| `4` | wave hello | Wave animation |
| `5` | sit down | Sit down |
| `6` | stand up | Stand up |
| `7` | what do you see | Describe visible objects |
| `8` | introduce yourself | Long animated introduction |
| `9` | dance | Dance sequence |
| `0` | come here | Walk forward |

### Simulation Control

| Key | Action | Description |
|-----|--------|-------------|
| `f` | Simulate robot fall | Triggers NAO hardware fall event (ALMemory) |
| `p` | Simulate person fall | Injects fallen pose into fall monitor (vision-based) |
| `r` | Recover from fall | Clears both robot and person fall states |
| `d` | Simulate disconnect | Stops heartbeat to trigger watchdog |
| `q` | Quit | Graceful shutdown |

### Notes on Hotkeys

- Hotkeys bypass the full voice pipeline (wake word + STT + verification)
- The command is injected directly into the FSM at the EXECUTING state
- This is equivalent to a verified owner giving the command
- Useful for quick demos without voice setup

---

## 9. Demo Scenarios

These are recommended sequences to demonstrate the system's capabilities.

### Demo 1: Face Tracking (Core Feature)

1. Start: `python run_simulation.py` (or press `1` for hotkey)
2. Say "hey nao" → "follow me"
3. Move your face left/right/up/down in front of the camera
4. Observe:
   - Green box follows your face in the camera panel
   - Head yaw/pitch values change in the state panel
   - `move_head` commands stream in the command log (15 Hz)
   - If you move far enough, `walk_toward` body realignment appears
5. Say "hey nao" → "stop" to end tracking

### Demo 2: Object Detection

1. Place objects in front of the camera (phone, bottle, cup)
2. Say "hey nao" → "find my phone" (or press `3`)
3. Observe:
   - YOLO detection boxes appear on camera (blue rectangles)
   - Head scans through angles [0, 0.5, -0.5, 1.0, -1.0]
   - NAO announces when found or timeout
4. Say "hey nao" → "what do you see" (or press `7`)
5. NAO describes all visible objects

### Demo 3: Parallel Execution

1. Press `1` (follow me) → face tracking starts
2. While tracking, press `4` (wave hello)
3. Observe: wave animation runs on ARMS channel while HEAD tracking continues
4. The state panel shows LEGS=idle, SPEECH=speaking, ARMS=animating simultaneously

### Demo 4: State Guards

1. Press `5` (sit down) → robot sits
2. Press `1` (follow me) → observe: "Let me stand up first" → auto-stands → then follows
3. This demonstrates the `_ensure_standing()` guard from Phase 3

### Demo 5: Robot Fall Detection

1. Start face tracking (press `1`)
2. Press `f` → NAO robot fall event triggers
3. Observe: servo stops, "I have fallen! Please help me up." in log
4. Press `r` → fall cleared
5. Press `6` → stand up, press `1` → resume tracking

### Demo 5b: Person Fall Detection (Improvement 4)

1. The fall monitor starts automatically at boot (no need for "follow me")
2. Observe the state panel: "Fall: MONITORING" (green) or "Fall: CALIBRATING" (yellow)
3. Press `p` → simulates a person falling (injects fallen pose data)
4. Observe:
   - Skeleton overlay on camera panel turns **RED**
   - Flashing **"FALL DETECTED"** banner on camera panel
   - Fall state changes to **TRIGGERED** (red flashing dot)
   - Console: `[SYS] PERSON FALL DETECTED!` (red text)
   - Robot says "Are you okay? I think you may have fallen!"
   - After 10 seconds with no response: "I'm calling for help!"
5. Press `r` → clears fall, detector re-arms to MONITORING
6. Note: this works in ANY state (IDLE, during a command, during YOLO search)

### Demo 6: Watchdog Safety

1. Press `d` → simulate disconnect (heartbeat stops)
2. Wait 10 seconds
3. Observe: watchdog triggers, robot sits, "Lost connection" message
4. Shows the safety mechanism for when the RPi brain crashes

### Demo 7: Full Voice Pipeline

1. Start with full voice mode: `python run_simulation.py`
2. Say "hey nao"
3. Observe: NAO says "Yes?" (visible in speech channel + audible via PC TTS)
4. Say "introduce yourself"
5. Observe: long animated speech, SPEECH channel busy
6. While NAO is speaking, say "hey nao" → "wave hello"
7. Observe: wave runs in parallel with remaining speech

### Demo 8: Object Retrieval (Bring Me)

1. Say "hey nao" → "find my phone" (or press `3` for find)
2. Wait for NAO to locate the phone
3. Say "hey nao" → "bring me my phone"
4. Observe the 5-phase sequence:
   - **Phase 1:** Validates last found object (must be recent)
   - **Phase 2:** Approaches the object (walks toward it)
   - **Phase 3:** Pickup (crouch → reach down → close hand → stand up)
   - **Phase 4:** Finds person (uses full-follow servo + person search if needed)
   - **Phase 5:** Delivers (offers arm → waits → opens hand → rests arm)
5. Alternative: after finding an object, say "go to it" or "pick it up" to just approach and grab

---

## 10. Configuration

### sim_config.py — All Settings

The file `sim_config.py` contains all configurable parameters:

```python
# Speed control
SIM_SPEED_MULTIPLIER = 1.0    # 0.5 = 2x faster, 2.0 = 2x slower

# PC text-to-speech
SIM_PC_TTS = True             # Set False to disable speech output

# Skip verification
SIM_SKIP_VERIFY = False       # Set True to accept all speakers

# Camera settings
CAMERA_INDEX = 0              # Change if using external webcam
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Network (should not need to change)
NAO_IP = "127.0.0.1"
NAO_PORT = 5555
```

### Adjusting for Different PCs

**Low-end PC (slow CPU):**
```bash
python run_simulation.py --no-yolo --speed 0.5
```
Disables YOLO (which is CPU-intensive) and runs mock timings at 2x speed.

**No peripherals at all:**
```bash
python run_simulation.py --keyboard --no-camera --no-tts
```
Everything via keyboard, synthetic camera, no audio output.

**Multiple cameras:**
```bash
python run_simulation.py --camera 1
```
Use device index 1 (e.g., external USB webcam instead of built-in).

---

## 11. Troubleshooting

### "ModuleNotFoundError: No module named 'vosk'"

**Cause:** Dependencies not installed.
**Fix:** `pip install -r requirements_simulation.txt`

### "FileNotFoundError: models/vosk-model-small-en-us-0.15"

**Cause:** Vosk model not downloaded.
**Fix:** Download from https://alphacephei.com/vosk/models or use `--no-mic`.

### "Could not open camera"

**Cause:** No webcam available or wrong index.
**Fix:** Use `--no-camera` for synthetic mode, or `--camera N` with correct index.
To find available cameras:
```bash
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: available')
        cap.release()
"
```

### "No sound output"

**Cause:** `pyttsx3` not installed or audio device issue.
**Fix:** `pip install pyttsx3` or check system audio settings.
If pyttsx3 crashes, use `--no-tts` and watch the command log instead.

### "YOLO model download failed"

**Cause:** No internet connection for first-time download.
**Fix:** Download `yolov8n.pt` manually from:
https://github.com/ultralytics/assets/releases/
Place it in the working directory or `~/.ultralytics/`.

### Dashboard window is tiny / wrong resolution

**Cause:** Display scaling on high-DPI screens.
**Fix:** The dashboard is fixed at 1440x900. On high-DPI screens, the window
may appear small. Set Windows display scaling to 100% or adjust the
`DASHBOARD_WIDTH` / `DASHBOARD_HEIGHT` constants in `gui/dashboard.py`.

### Face detection not working with synthetic camera

**Cause:** The synthetic moving dot may not trigger MediaPipe reliably.
**Fix:** This is expected — the synthetic camera is primarily for testing
the servo loop and command pipeline. For face tracking demos, use a real webcam.

### "Connection refused" at startup

**Cause:** Mock server hasn't started yet.
**Fix:** The simulation waits up to 5 seconds for the server. If it still
fails, check if port 5555 is already in use:
```bash
# Windows:
netstat -aon | findstr 5555

# Linux/macOS:
lsof -i :5555
```

### Speaker verification always rejects

**Cause:** Using stub model (328 KB) instead of real model, or master
embedding is from a different model version.
**Fix:** Either:
- Use `--no-verify` to skip verification
- Export and deploy the real ECAPA model (see Section 3)
- Re-run enrollment after changing models

### High CPU usage

**Cause:** Face detection (MediaPipe) + YOLO running simultaneously.
**Fix:** Use `--no-yolo` if not demonstrating object detection. The face
tracker alone uses ~15% CPU on a modern processor.

---

## 12. How It Works

### What is Real vs Simulated

This simulation runs the **exact same code** as the real ElderGuard system,
with only two thin adapter layers:

**Real (unchanged production code):**
- Voice recognition (Vosk STT with grammar constraints)
- Speaker verification (ECAPA-TDNN via ONNX Runtime)
- Face detection (MediaPipe BlazeFace)
- Pose estimation (MediaPipe Pose for fall detection)
- Person fall detection (4-signal fusion, always-on 5 Hz monitor)
- PID visual servo controller (15 Hz tracking loop)
- Command parser (phrase → intent mapping)
- TCP client (JSON-over-TCP with two-phase ACK)
- FSM state machine (IDLE → LISTENING → VERIFYING → EXECUTING)
- NAO server dispatch, channel workers, state machine
- Object pickup sequence (arm control, grip, carry, offer, release)
- Watchdog, robot fall detection, heartbeat, auto-reconnect

**Adapted (thin wrappers):**
- Camera: `PcCamera` — same interface, no V4L2 flag (Windows-compatible)
- Object detector: `PcObjectDetector` — same interface, uses ultralytics instead of tflite-runtime

**Simulated (mock layer):**
- NAO motor proxies: `MockMotion` — tracks joint angles, simulates walk timing
- NAO speech: `MockTts` — simulates duration, optionally speaks via PC TTS
- NAO posture: `MockPosture` — tracks posture state, simulates transition time
- NAO memory: `MockMemory` — provides fall detection data

### Thread Architecture

The simulation runs ~12 concurrent threads, identical to the real system:

```
Main Thread (Dashboard GUI @ 30 FPS)
  |
  +-- Brain Thread (AssistantApp FSM)
  |     +-- Camera Thread (frame grabbing)
  |     +-- Servo Thread (15 Hz PID loop, only during follow-me)
  |     +-- Fall Monitor Thread (5 Hz pose + fall detection, always on)
  |     +-- TCP Reader Thread (response dispatch)
  |     +-- Heartbeat Thread (2s keepalive)
  |
  +-- Server Thread (TCP handler)
        +-- LEGS Worker (walk/pose/rest)
        +-- SPEECH Worker (say FIFO)
        +-- ARMS Worker (wave/dance)
        +-- Watchdog (connection monitor)
        +-- Robot Fall Detector (ALMemory polling)
```

### Network Architecture

Even though everything runs on one PC, the brain and server communicate
over a real TCP socket on `localhost:5555`:

```
Brain (TCP Client)                   Server (TCP Server)
  |                                    |
  |-- {"action":"say","text":"Hi"} --> |
  |                                    |-- [SPEECH worker executes]
  |<-- {"status":"accepted"} ---------|  (immediate ack)
  |                                    |
  |<-- {"status":"ok","type":"done"} -|  (when speech finishes)
```

This means the full protocol — newline-delimited JSON, fire-and-forget
mode, two-phase ACK, heartbeat keepalive, watchdog timeout — all work
exactly as they would between a real RPi and NAO robot.

### Session Logs

Every simulation session creates a JSONL log file in `logs/`:
```
logs/sim_20260312_143005.jsonl
```

Each line is a JSON event:
```json
{"ts": "2026-03-12T14:30:05.123", "event": "fsm_state_change", "from": "IDLE", "to": "LISTENING"}
{"ts": "2026-03-12T14:30:06.456", "event": "stt_result", "text": "follow me"}
{"ts": "2026-03-12T14:30:06.789", "event": "speaker_verify", "score": 0.82, "accepted": true}
{"ts": "2026-03-12T14:30:06.810", "event": "tcp_command", "action": "say", "text": "Okay, I will follow you."}
```

These logs can be used for:
- Debugging issues
- Analyzing system performance
- Presenting metrics in reports
- Verifying the pipeline works correctly
