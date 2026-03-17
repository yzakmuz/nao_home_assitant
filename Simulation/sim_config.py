"""
sim_config.py -- PC simulation overrides for settings.py.

This module is installed as sys.modules["settings"] by bootstrap.py,
so every `from settings import ...` in the real rpi_brain code resolves here.

We import everything from the real settings.py first, then override
PC-specific values.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Helper: convert a path to Windows short (8.3) form so that C libraries
# (like Vosk / KaldiRecognizer) that cannot handle Unicode paths still work.
# ---------------------------------------------------------------------------
def _short_path(long_path):
    """Return the Windows 8.3 short path for *long_path*, or the original
    path unchanged on non-Windows / if the conversion fails."""
    if sys.platform != "win32":
        return long_path
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(512)
        rv = ctypes.windll.kernel32.GetShortPathNameW(long_path, buf, 512)
        if rv and rv < 512:
            return buf.value
    except Exception:
        pass
    return long_path

# ---------------------------------------------------------------------------
# Compute paths relative to this file
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)  # ElderGuard-Humanoid_Assistant_Robot/
_RPI_BRAIN_DIR = os.path.join(_REPO_ROOT, "nao_assistant", "rpi_brain")

# ---------------------------------------------------------------------------
# Import ALL real settings as baseline (we'll override what's needed below)
# ---------------------------------------------------------------------------
# We can't do `from settings import *` because *we* are settings.
# Instead, manually exec the real settings.py into our namespace.
_real_settings_path = os.path.join(_RPI_BRAIN_DIR, "settings.py")
with open(_real_settings_path, "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), _real_settings_path, "exec"))

# ---------------------------------------------------------------------------
# Network -- localhost (mock NAO server runs on same PC)
# ---------------------------------------------------------------------------
NAO_IP = "127.0.0.1"
NAO_PORT = 5555

# ---------------------------------------------------------------------------
# Camera -- no V4L2, higher resolution for PC
# ---------------------------------------------------------------------------
CAMERA_INDEX = int(os.getenv("SIM_CAMERA_INDEX", "0"))
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ---------------------------------------------------------------------------
# Audio -- use system default mic, auto-detect native sample rate
# ---------------------------------------------------------------------------
MIC_DEVICE_INDEX = None
try:
    import sounddevice as _sd
    _dev = _sd.query_devices(kind="input")
    MIC_NATIVE_RATE = int(_dev["default_samplerate"])
except Exception:
    pass  # keep the value inherited from real settings.py (44100)

# ---------------------------------------------------------------------------
# Model paths -- resolve relative to rpi_brain (where real code expects them)
# ---------------------------------------------------------------------------
VOSK_MODEL_PATH = _short_path(os.getenv(
    "VOSK_MODEL_PATH",
    os.path.join(_RPI_BRAIN_DIR, "models", "vosk-model-small-en-us-0.15"),
))
SPEAKER_MODEL_PATH = _short_path(os.getenv(
    "SPEAKER_MODEL_PATH",
    os.path.join(_RPI_BRAIN_DIR, "models", "ecapa_tdnn.onnx"),
))
MASTER_EMBEDDING_PATH = _short_path(os.getenv(
    "MASTER_EMBEDDING_PATH",
    os.path.join(_RPI_BRAIN_DIR, "models", "master_embedding.npy"),
))
YOLO_MODEL_PATH = _short_path(os.getenv(
    "YOLO_MODEL_PATH",
    os.path.join(_RPI_BRAIN_DIR, "models", "yolov8n.tflite"),
))

# ---------------------------------------------------------------------------
# Memory thresholds -- PC has plenty of RAM
# ---------------------------------------------------------------------------
RAM_WARNING_THRESHOLD_MB = 500
RAM_CRITICAL_THRESHOLD_MB = 200

# ---------------------------------------------------------------------------
# Simulation-specific settings
# ---------------------------------------------------------------------------
SIM_SPEED_MULTIPLIER = float(os.getenv("SIM_SPEED_MULTIPLIER", "1.0"))
SIM_PC_TTS = os.getenv("SIM_PC_TTS", "true").lower() in ("1", "true", "yes")
SIM_SKIP_VERIFY = False
SIM_USE_SPEECHBRAIN_VERIFY = True  # Use SpeechBrain (PyTorch) instead of ONNX stub
SIM_NO_MIC = False
SIM_NO_CAMERA = False
SIM_NO_YOLO = False
SIM_FALL_DETECTION = True           # Enable person fall detection
SIM_FALL_POSE_OVERLAY = True        # Draw skeleton overlay on camera panel
SIM_LOG_DIR = os.path.join(_THIS_DIR, "logs")
