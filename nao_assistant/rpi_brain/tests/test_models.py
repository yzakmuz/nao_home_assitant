#!/usr/bin/env python3
"""test_models.py — Verify all models load and run on RPi."""
import sys, os, time
import numpy as np

print("=" * 50)
print("  RPI Brain — Model Loading Test")
print("=" * 50)

# ── 1. Vosk STT ──
print("\n[1/4] Vosk STT...")
try:
    from vosk import Model, KaldiRecognizer
    t = time.time()
    model = Model("models/vosk-model-small-en-us-0.15")
    grammar = '["hey nao", "follow me", "stop", "sit down", "stand up", "dance", "wave", "rest", "find my keys", "[unk]"]'
    rec = KaldiRecognizer(model, 16000, grammar)
    # Feed 1 second of silence to test
    silence = np.zeros(16000, dtype=np.int16).tobytes()
    rec.AcceptWaveform(silence)
    print(f"  ✓ Loaded in {time.time()-t:.1f}s")
    print(f"  ✓ Grammar recognizer works")
    del rec, model
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# ── 2. ECAPA-TDNN (Speaker Verification) ──
print("\n[2/4] ECAPA-TDNN (ONNX speaker verify)...")
try:
    import onnxruntime as ort
    t = time.time()
    sess = ort.InferenceSession("models/ecapa_tdnn.onnx")
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    print(f"  ✓ Loaded in {time.time()-t:.1f}s")
    print(f"  Inputs:  {[(i.name, i.shape) for i in inputs]}")
    print(f"  Outputs: {[(o.name, o.shape) for o in outputs]}")
    # Try a dummy inference
    dummy_shape = [1 if isinstance(d, str) else d for d in inputs[0].shape]
    dummy = np.random.randn(*dummy_shape).astype(np.float32)
    t = time.time()
    result = sess.run(None, {inputs[0].name: dummy})
    print(f"  ✓ Inference OK in {time.time()-t:.3f}s")
    print(f"  Embedding shape: {result[0].shape}")
    del sess
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# ── 3. MediaPipe BlazeFace ──
print("\n[3/4] MediaPipe BlazeFace...")
try:
    import mediapipe as mp
    t = time.time()
    fd = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
    # Feed a dummy 320x240 RGB frame
    dummy_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    result = fd.process(dummy_frame)
    print(f"  ✓ Loaded + inference in {time.time()-t:.1f}s")
    print(f"  Detections on noise: {result.detections is not None}")
    fd.close()
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# ── 4. YOLOv8n TFLite ──
print("\n[4/4] YOLOv8n TFLite (INT8)...")
try:
    import tflite_runtime.interpreter as tflite
    t = time.time()
    interp = tflite.Interpreter(model_path="models/yolov8n_int8.tflite")
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()
    print(f"  ✓ Loaded in {time.time()-t:.1f}s")
    print(f"  Input:  {inp_det['shape']} dtype={inp_det['dtype']}")
    print(f"  Outputs: {len(out_det)} tensors")
    # Dummy inference
    dummy = np.random.randint(0, 255, inp_det['shape'], dtype=inp_det['dtype'])
    interp.set_tensor(inp_det['index'], dummy)
    t = time.time()
    interp.invoke()
    print(f"  ✓ Inference OK in {time.time()-t:.3f}s")
    for i, o in enumerate(out_det):
        tensor = interp.get_tensor(o['index'])
        print(f"  Output[{i}]: shape={tensor.shape}")
    del interp
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# ── RAM Report ──
print("\n" + "=" * 50)
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"  RAM: {mem.used/1024**2:.0f}MB used / {mem.total/1024**2:.0f}MB total ({mem.percent}%)")
except:
    pass
print("=" * 50)
print("  DONE")
