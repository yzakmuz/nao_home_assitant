# NAO Autonomous Personal Assistant — Offline Architecture

## System Overview

A fully offline, dual-device autonomous robot assistant. A **Raspberry Pi 4** (the Brain)
performs all AI inference and decision-making, then sends simple JSON commands over TCP
to a **NAO V5 Robot** (the Body), which executes physical actions via its native NAOqi API.

```
┌─────────────────────────────────────────────────────────┐
│  Raspberry Pi 4 "The Brain" (Python 3.9+)               │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐    │
│  │ USB Mic  │→ │ Vosk STT │→ │ Speaker Verify     │    │
│  │ Stream   │  │ (Grammar)│  │ (ECAPA-TDNN ONNX)  │    │
│  └──────────┘  └──────────┘  └────────┬───────────┘    │
│                                       ↓                 │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐    │
│  │ Pi Cam   │→ │MediaPipe │→ │ Visual Servo       │──┐ │
│  │ Stream   │  │BlazeFace │  │ Controller (PID)   │  │ │
│  └──────────┘  └──────────┘  └────────────────────┘  │ │
│                                                       │ │
│  ┌──────────────────┐  ┌──────────────────────────┐  │ │
│  │ YOLOv8n (Dynamic)│  │ Command Parser / FSM     │  │ │
│  │ Load/Unload RAM  │  │ (Intent → Action)        │  │ │
│  └──────────────────┘  └────────────┬─────────────┘  │ │
│                                     ↓                 ↓ │
│                          ┌─────────────────────┐        │
│                          │  TCP Client (JSON)  │        │
│                          └─────────┬───────────┘        │
└────────────────────────────────────┼────────────────────┘
                    Ethernet TCP     │
┌────────────────────────────────────┼────────────────────┐
│  NAO V5 "The Body" (Python 2.7)   │                     │
│                          ┌─────────┴───────────┐        │
│                          │  TCP Server (JSON)  │        │
│                          └─────────┬───────────┘        │
│                                    ↓                    │
│                          ┌─────────────────────┐        │
│                          │  NAOqi Dispatcher   │        │
│                          │  ALMotion, ALTts,   │        │
│                          │  ALAnimatedSpeech   │        │
│                          └─────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

## Data Flow: Wake Word → Action

1. `mic_stream` captures 16kHz mono audio in chunks.
2. `stt_engine` (Vosk) performs grammar-constrained recognition.
   - If wake word ("hey nao") is detected → transition to LISTENING state.
3. Full utterance captured → `speaker_verify` computes ECAPA-TDNN embedding.
   - Cosine similarity vs. enrolled master embedding → accept/reject.
4. Accepted → `command_parser` extracts intent (e.g., "find my keys", "follow me").
5. `main.py` FSM dispatches:
   - **Motion commands** → JSON sent via `tcp_client` → NAO moves/speaks.
   - **"follow me"** → enables `visual_servo` face-tracking loop.
   - **"find <object>"** → dynamically loads YOLO, searches, unloads.

## Directory Structure Rationale

```
nao_assistant/
├── nao_body/                  # ── Deployed to NAO Robot ──
│   ├── server.py              # TCP server + NAOqi command dispatcher
│   └── motion_library.py      # Reusable motion presets (poses, gaits)
│
├── rpi_brain/                 # ── Deployed to Raspberry Pi ──
│   ├── main.py                # Entry point: state machine orchestrator
│   ├── settings.py            # ALL tunables (IPs, thresholds, paths)
│   ├── requirements.txt       # Frozen pip dependencies
│   │
│   ├── comms/                 # Network boundary layer
│   │   └── tcp_client.py      # Thread-safe JSON-over-TCP client
│   │
│   ├── audio/                 # Audio perception pipeline
│   │   ├── mic_stream.py      # Ring-buffer microphone capture
│   │   ├── stt_engine.py      # Vosk grammar-based recognition
│   │   └── speaker_verify.py  # ECAPA-TDNN ONNX cosine verify
│   │
│   ├── vision/                # Visual perception pipeline
│   │   ├── camera.py          # Pi Camera V4L2 thread-safe capture
│   │   ├── face_tracker.py    # MediaPipe BlazeFace detector
│   │   └── object_detector.py # YOLOv8n dynamic load/unload
│   │
│   ├── servo/                 # Control systems
│   │   └── visual_servo.py    # PID head tracking + body realignment
│   │
│   ├── command/               # NLU / intent layer
│   │   └── parser.py          # Grammar → intent → action mapping
│   │
│   ├── utils/                 # Cross-cutting concerns
│   │   └── memory.py          # RSS monitoring, GC triggers
│   │
│   └── models/                # Git-ignored model binaries
│       ├── vosk-model-small-en-us-0.15/
│       ├── ecapa_tdnn.onnx
│       ├── master_embedding.npy
│       └── yolov8n.tflite
│
└── README.md
```

**Why this split?**
- `nao_body/` is a self-contained Python 2.7 package that touches `naoqi`.
  It never imports anything from `rpi_brain/`. Deploy it to the NAO independently.
- `rpi_brain/` is a pure Python 3.9+ package. It has zero knowledge of NAOqi.
  The only coupling is the JSON protocol defined implicitly by `tcp_client.py`
  and `server.py`.
- `models/` is separated so large binaries stay out of version control.

## Setup

### NAO Side
```bash
scp -r nao_body/ nao@<NAO_IP>:~/assistant/
ssh nao@<NAO_IP>
cd ~/assistant && python server.py
```

### RPi Side
```bash
cd rpi_brain/
pip install -r requirements.txt
# Place models in ./models/
python main.py
```
