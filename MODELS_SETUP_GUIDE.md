# NAO Robot Brain — Models Setup Summary

**Status: ✓ PHASE A COMPLETE — Ready for end-to-end testing**

---

## Models Directory Contents

```
rpi_brain/models/
├── vosk-model-small-en-us-0.15/        (dir)     ~65 MB  ✓ STT engine
├── ecapa_tdnn.onnx                      (stub)    328 KB  ✓ Speaker verification  
└── yolov8n_int8.tflite                           3.3 MB  ✓ Object detection
```

### **What's Ready for Testing (Phase A)**

| Component | Model | Status | Notes |
|-----------|-------|--------|-------|
| Speech-to-text (STT) | Vosk small-en-us-0.15 | ✓ Live | Downloaded from alphacephei.com; optimized for RPi |
| Object Detection | YOLOv8n INT8 | ✓ Live | Quantized TFLite; runs on 2GB RAM |
| Speaker Verification | ECAPA-TDNN (stub) | ✓ Stub | Lightweight ~328 KB placeholder; loads without errors |
| Face Tracking | MediaPipe BlazeFace | ✓ Built-in | Loaded dynamically by vision/face_tracker.py |

---

## Phase A: End-to-End Testing

The **stub ECAPA-TDNN** model is a minimal 328 KB ONNX file that:
- ✓ Has correct input/output signatures (audio waveform → 256-dim embedding)
- ✓ Loads successfully with onnxruntime
- ✓ Produces deterministic outputs for testing
- ✓ Allows TCP communication & pipeline validation
- ✗ Does NOT perform real speaker verification (intentional)

### What You Can Test Now
1. **Vosk STT Pipeline**
   - Wake word detection ("hey nao")
   - Grammar-constrained speech recognition
   - Command parsing

2. **Vision Pipeline**
   - Face detection & tracking (MediaPipe)
   - Object detection with YOLO
   - Visual servo integration

3. **TCP Communication**
   - RPi client → NAO server messages
   - Command execution on NAO body
   - Response handling

4. **FSM Flow**
   - State transitions (IDLE → LISTENING → VERIFYING → EXECUTING)
   - Error recovery
   - Timeout handling

### What Will NOT Work Yet
- **Speaker verification** (stub returns dummy embeddings)
- Real authentication of commands
- Distinguish between authorized/unauthorized speakers

---

## Phase B: Dynamic Memory Management (Future)

Once testing is complete, implement dynamic load/unload for speaker verification to fit on 2GB RPi:

### Current Architecture (Memory Problem)
```python
class SpeakerVerifier:
    def __init__(self):
        self._session = ort.InferenceSession(SPEAKER_MODEL_PATH)  # ~20-30 MB resident
        # Model stays in RAM for duration of app
```

**Memory Profile on RPi 4 (2GB):**
- Vosk STT: ~100 MB (unpacked)
- MediaPipe: ~40 MB
- YOLO: ~10 MB (quantized)
- ECAPA-TDNN: ~20-30 MB (resident) ← **Problem**
- System/other: ~300 MB
- **Total: ~500 MB overhead — leaves <1.5 GB for operations**

### Proposed Solution (Phase B)

Modify `audio/speaker_verify.py` to implement context manager pattern:

```python
# Phase B: Dynamic Load/Unload Pattern

class SpeakerVerifier:
    def __init__(self):
        self._session = None  # Don't load in __init__
        self._master_emb = ...
    
    def load_model(self):
        """Load ONNX model into RAM."""
        self._session = ort.InferenceSession(SPEAKER_MODEL_PATH)
    
    def unload_model(self):
        """Free ONNX model from RAM and force garbage collection."""
        self._session = None
        gc.collect()
    
    def verify(self, audio):
        """Load model, verify, then unload."""
        self.load_model()
        try:
            emb = self.compute_embedding(audio)
            score = cosine_similarity(emb, self._master_emb)
            return score >= THRESHOLD
        finally:
            self.unload_model()  # Always cleanup
```

**Usage in `main.py`:**
```python
def _handle_verifying_state(self):
    # ... audio capture code ...
    is_valid, score = self._speaker.verify(audio)  # Load/unload happens inside
    # Model only resident during the 1-2 second verification window
```

**Expected RAM savings:**
- ~20-30 MB freed after each command
- Model only loaded for ~2-3 seconds per command
- Leaves ~200+ MB buffer for other operations
- Safe for continuous 24/7 operation on 2GB RPi

---

## Integration Checklist for Phase B

When ready to implement dynamic loading:

- [ ] Update `audio/speaker_verify.py`:
  - [ ] Move ORT session initialization from `__init__` to new `load_model()`
  - [ ] Implement `unload_model()` with `gc.collect()`
  - [ ] Wrap `compute_embedding()` calls with load/unload

- [ ] Update `main.py`:
  - [ ] Call `self._speaker.verify()` in `_handle_verifying_state()`
  - [ ] Verify memory metrics with `utils.memory.log_memory()`
  - [ ] Test OOM conditions on RPi 4

- [ ] Testing:
  - [ ] Monitor RAM usage during continuous commands
  - [ ] Test 100+ commands without restart
  - [ ] Verify enrollment (`enroll_speaker.py`) still works
  - [ ] Test with real ECAPA-TDNN model once obtained

---

## Real ECAPA-TDNN Model (When Needed)

When you're ready to replace the stub with the real model:

1. **Obtain the model:**
   ```bash
   # Option 1: Manual download from Hugging Face
   # https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
   # Download "model.onnx" and rename to "ecapa_tdnn.onnx"
   
   # Option 2: Export from SpeechBrain (requires PyTorch)
   from speechbrain.pretrained import EncoderClassifier
   classifier = EncoderClassifier.from_hparams(
       source="speechbrain/spkrec-ecapa-voxceleb"
   )
   torch.onnx.export(..., "ecapa_tdnn.onnx")
   ```

2. **Replace the stub:**
   ```bash
   cp /path/to/real/ecapa_tdnn.onnx models/ecapa_tdnn.onnx
   ```

3. **No code changes needed** — module interface is identical

4. **Run enrollment:**
   ```bash
   python enroll_speaker.py --duration 5
   # This creates models/master_embedding.npy
   ```

---

## Current Testing Ready ✓

**All components needed for Phase A testing are in place:**
- ✓ Vosk speech recognition (optimized for RPi)
- ✓ MediaPipe vision (lightweight BlazeFace)
- ✓ YOLOv8n object detection (INT8 quantized)
- ✓ ECAPA-TDNN stub (allows verification testing flow)
- ✓ TCP client ready for NAO communication
- ✓ FSM orchestrator ready

You can now run end-to-end tests without OOM concerns!

---

## Files Generated

- `models/ecapa_tdnn.onnx` — Stub model (328 KB)
- This document — Planning guide for Phase B

Temporary scripts removed (cleanup done).

