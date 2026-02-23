#!/usr/bin/env python3
"""
QUICK START: End-to-End Testing with Stub Models (Phase A)

This is a ready-to-test setup. All models are in place:
  ✓ Vosk STT (65 MB)
  ✓ ECAPA-TDNN (stub, 328 KB)
  ✓ YOLOv8n (3.3 MB)
  ✓ MediaPipe (built-in)

Next Steps:
  1. Review this checklist
  2. Start testing the FSM on your RPi 4
  3. Once validated → Move to Phase B (dynamic memory management)
"""

print(__doc__)

# ============================================================================
# PHASE A: TESTING CHECKLIST
# ============================================================================

TESTING_CHECKLIST = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PHASE A: SYSTEM VALIDATION                           │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: Verifying Model Setup (Should Already Pass)
─────────────────────────────────────────────────────
  [ ] Vosk model exists:        rpi_brain/models/vosk-model-small-en-us-0.15/
  [ ] ECAPA stub exists:        rpi_brain/models/ecapa_tdnn.onnx (328 KB)
  [ ] YOLO model exists:        rpi_brain/models/yolov8n_int8.tflite (3.3 MB)
  
  Test: python -c "from audio.stt_engine import SttEngine; print('✓ Vosk loads')"
        python -c "from audio.speaker_verify import SpeakerVerifier; print('✓ ECAPA loads')"
        python -c "from vision.object_detector import ObjectDetector; print('✓ YOLO loads')"

STEP 2: Testing Individual Components
──────────────────────────────────────
  [ ] Microphone stream captures audio (mic_stream.py)
  [ ] STT recognizes wake word "hey nao" (stt_engine.py)
  [ ] Face detection finds faces (face_tracker.py)
  [ ] YOLO detects objects (object_detector.py)
  [ ] TCP client connects to NAO (tcp_client.py)

  Run: Script coming soon to test each module independently

STEP 3: Testing FSM Pipeline (Full Integration)
────────────────────────────────────────────────
  [ ] Start in IDLE, track faces
  [ ] Detect wake word → LISTENING state
  [ ] Capture command → VERIFYING state
  [ ] Speaker verification (stub) → EXECUTING state
  [ ] Send command to NAO body over TCP
  [ ] Return to IDLE
  
  Run: python main.py

STEP 4: Memory & Performance Monitoring
────────────────────────────────────────
  [ ] Monitor RAM usage: watch "free -h" in one terminal
  [ ] Monitor logs: tail -f output.log
  [ ] Run for 10+ commands without crashes
  [ ] Check memory doesn't exceed 1.5 GB during operation
  
  Memory expected:
    - Idle: ~500-600 MB
    - After STT: +50-100 MB
    - During YOLO search: +50 MB
    - Return to idle: Back to ~500 MB

STEP 5: TCP Communication Test
───────────────────────────────
  [ ] NAO body server running (on NAO robot)
  [ ] RPi connects to NAO (check logs for "Connected to NAO")
  [ ] Commands reach NAO (monitor NAO's server output)
  [ ] NAO responds with motion (wave, say hello, move head)
  
  Commands to test:
    "hey nao, wave hello"
    "hey nao, find my keys"
    "hey nao, what do you see"
    "hey nao, stand up"

STEP 6: Stress Testing (Before Production)
───────────────────────────────────────────
  [ ] Run 50+ commands in sequence
  [ ] Check for memory leaks (RAM should not grow unbounded)
  [ ] Verify no zombie processes from YOLO/MediaPipe
  [ ] Test error recovery (disconnect/reconnect NAO)
  [ ] Test timeout behavior (NAO unreachable)

─────────────────────────────────────────────────────────────────────────────
PHASE A SUCCESS CRITERIA:
  ✓ FSM completes at least 20 command cycles
  ✓ RAM usage stays below 1.5 GB
  ✓ TCP communication is stable
  ✓ No Python crashes or hangs
  ✓ Commands execute on NAO robot
─────────────────────────────────────────────────────────────────────────────
"""

print(TESTING_CHECKLIST)

# ============================================================================
# PHASE B: MEMORY OPTIMIZATION (Manual Implementation Needed)
# ============================================================================

PHASE_B_NOTES = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PHASE B: DYNAMIC MEMORY MANAGEMENT                        │
│                          (After Phase A Success)                            │
└─────────────────────────────────────────────────────────────────────────────┘

When Phase A testing is complete and stable, implement Option B:

MODIFICATIONS NEEDED:

1. audio/speaker_verify.py
   ─────────────────────────
   [ ] Move ORT session initialization from __init__ to load_model()
   [ ] Add unload_model() method that frees the session and calls gc.collect()
   [ ] Update verify() / compute_embedding() to wrap with load/unload pattern
   
   Pseudocode:
     def verify(self, audio):
         self.load_model()
         try:
             embedding = self.compute_embedding(audio)
             score = cosine_similarity(embedding, self._master_emb)
             return score >= THRESHOLD
         finally:
             self.unload_model()  # Always cleanup

2. main.py (FSM orchestrator)
   ──────────────────────────
   [ ] In _handle_verifying_state(), call self._speaker.verify() normally
       (load/unload happens inside that method)
   [ ] Add memory logging: utils.memory.log_memory() after each state

3. Testing Phase B
   ────────────────
   [ ] Run 200+ commands without restart
   [ ] Monitor RAM growth (should be <50 MB variance)
   [ ] Test with real ECAPA-TDNN model (once obtained)
   [ ] Verify enrollment still works: python enroll_speaker.py

EXPECTED RESULTS:
  Before Phase B: 500-700 MB RAM during operation
  After Phase B:  300-500 MB RAM (constant, no growth)
  
  This gives you 1.5+ GB buffer on the 2 GB RPi for:
  - OS operations
  - System services
  - Emergency processes
  - Safe continuous operation

─────────────────────────────────────────────────────────────────────────────
"""

print(PHASE_B_NOTES)

# ============================================================================
# USEFUL COMMANDS FOR TESTING
# ============================================================================

USEFUL_COMMANDS = """
USEFUL COMMANDS FOR TESTING ON RASPBERRY PI:
─────────────────────────────────────────────

# Monitor memory in real-time
watch -n 1 free -h

# Monitor process memory
watch -n 1 "ps aux | sort -k6 -h | tail -15"

# Run main.py with logging
python main.py 2>&1 | tee assistant.log

# Check Python memory usage
python -c "import psutil; p=psutil.Process(); print(f'Memory: {p.memory_info().rss/1024**2:.1f} MB')"

# Test specific module loads
python -c "
from audio.mic_stream import MicStream
from audio.stt_engine import SttEngine
from audio.speaker_verify import SpeakerVerifier
from vision.camera import Camera
from vision.face_tracker import FaceTracker
from vision.object_detector import ObjectDetector
print('✓ All modules load successfully')
"

# Quick model inference test
python -c "
import numpy as np
from vision.object_detector import ObjectDetector
with ObjectDetector() as det:
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    results = det.detect(dummy_frame)
    print(f'✓ YOLO inference works: {len(results.detections)} detections')
"

# Test TCP client connectivity
python -c "
from comms.tcp_client import NaoTcpClient
tcp = NaoTcpClient()
tcp.connect()
if tcp.is_connected:
    print('✓ Connected to NAO')
else:
    print('✗ Cannot connect to NAO (check IP/network)')
"

─────────────────────────────────────────────────────────────────────────────
"""

print(USEFUL_COMMANDS)

print("\n" + "="*80)
print("READY FOR PHASE A TESTING")
print("="*80)
print("\nNext: Deploy to Raspberry Pi 4 and follow the testing checklist above.")
print("Documentation: See MODELS_SETUP_GUIDE.md for detailed architecture notes.")
