#!/usr/bin/env python3
"""test_models2.py — Re-verify all 4 models after onnxruntime upgrade."""
import sys, time
import numpy as np

print("=" * 50)
print("  RPI Brain — Model Test (Round 2)")
print("=" * 50)

# 1. Vosk
print("\n[1/4] Vosk STT...")
try:
    from vosk import Model, KaldiRecognizer
    t = time.time()
    model = Model("models/vosk-model-small-en-us-0.15")
    grammar = '["hey nao", "follow me", "stop", "sit down", "stand up", "dance", "wave", "rest", "find my keys", "[unk]"]'
    rec = KaldiRecognizer(model, 16000, grammar)
    silence = np.zeros(16000, dtype=np.int16).tobytes()
    rec.AcceptWaveform(silence)
    print(f"  ✓ OK — {time.time()-t:.1f}s")
    del rec, model
except Exception as e:
    print(f"  ✗ {e}")

# 2. ECAPA-TDNN
print("\n[2/4] ECAPA-TDNN (ONNX)...")
try:
    import onnxruntime as ort
    print(f"  onnxruntime version: {ort.__version__}")
    t = time.time()
    sess = ort.InferenceSession("models/ecapa_tdnn.onnx")
    inp = sess.get_inputs()[0]
    print(f"  ✓ Loaded in {time.time()-t:.1f}s")
    print(f"  Input: name={inp.name}, shape={inp.shape}")
    dummy_shape = [1 if isinstance(d, str) else d for d in inp.shape]
    dummy = np.random.randn(*dummy_shape).astype(np.float32)
    t = time.time()
    result = sess.run(None, {inp.name: dummy})
    print(f"  ✓ Inference OK in {time.time()-t:.3f}s")
    print(f"  Embedding shape: {result[0].shape}")
    del sess
except Exception as e:
    print(f"  ✗ {e}")

# 3. MediaPipe
print("\n[3/4] MediaPipe BlazeFace...")
try:
    import mediapipe as mp
    t = time.time()
    fd = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    dummy = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    fd.process(dummy)
    print(f"  ✓ OK — {time.time()-t:.1f}s")
    fd.close()
except Exception as e:
    print(f"  ✗ {e}")

# 4. YOLOv8n TFLite (fixed)
print("\n[4/4] YOLOv8n TFLite...")
try:
    import tflite_runtime.interpreter as tflite
    t = time.time()
    interp = tflite.Interpreter(model_path="models/yolov8n_int8.tflite")
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    print(f"  ✓ Loaded in {time.time()-t:.1f}s")
    print(f"  Input: {inp_det['shape']} dtype={inp_det['dtype']}")
    # Fixed: use correct dtype for dummy input
    dummy = np.random.rand(*inp_det['shape']).astype(inp_det['dtype'])
    interp.set_tensor(inp_det['index'], dummy)
    t = time.time()
    interp.invoke()
    out = interp.get_output_details()
    print(f"  ✓ Inference OK in {time.time()-t:.3f}s")
    for i, o in enumerate(out):
        print(f"  Output[{i}]: {interp.get_tensor(o['index']).shape}")
    del interp
except Exception as e:
    print(f"  ✗ {e}")

print("\n" + "=" * 50)
import psutil
mem = psutil.virtual_memory()
print(f"  RAM: {mem.used/1024**2:.0f}MB / {mem.total/1024**2:.0f}MB ({mem.percent}%)")
print("  ALL MODELS TESTED")
print("=" * 50)
