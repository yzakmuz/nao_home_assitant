"""
settings.py — Single source of truth for every tunable parameter.

Edit this file (or override via environment variables) instead of
touching engine code. All thresholds, paths, and network addresses live here.
"""

import os

# ---------------------------------------------------------------------------
# Network — TCP socket between RPi (client) and NAO (server)
# ---------------------------------------------------------------------------
NAO_IP = os.getenv("NAO_IP", "10.0.0.10")
NAO_PORT = int(os.getenv("NAO_PORT", "5555"))
TCP_TIMEOUT_S = float(os.getenv("TCP_TIMEOUT_S", "5.0"))
TCP_RECONNECT_DELAY_S = float(os.getenv("TCP_RECONNECT_DELAY_S", "3.0"))
TCP_BUFFER_SIZE = int(os.getenv("TCP_BUFFER_SIZE", "4096"))
MSG_DELIMITER = b"\n"  # newline-delimited JSON framing

# ---------------------------------------------------------------------------
# Audio — Microphone & Vosk STT
# ---------------------------------------------------------------------------
MIC_SAMPLE_RATE = 16000
MIC_CHANNELS = 1
MIC_CHUNK_FRAMES = 4000        # 250 ms per chunk at 16 kHz
MIC_DEVICE_INDEX = None         # None = system default; set int to override

VOSK_MODEL_PATH = os.getenv(
    "VOSK_MODEL_PATH", "models/vosk-model-small-en-us-0.15"
)

# Grammar-constrained phrases Vosk will listen for.
# The wake word MUST be the first entry.
WAKE_WORD = "hey nao"
GRAMMAR_PHRASES = [
    "hey nao",
    "follow me",
    "stop",
    "find my keys",
    "find my phone",
    "find my bottle",
    "find my cup",
    "wave hello",
    "say hello",
    "sit down",
    "stand up",
    "what do you see",
    "look left",
    "look right",
    "turn around",
    "come here",
    "introduce yourself",
    "dance",
    "[unk]",  # Vosk unknown-word bucket
]

# Maximum seconds to wait for a command after wake word
LISTEN_TIMEOUT_S = float(os.getenv("LISTEN_TIMEOUT_S", "5.0"))

# ---------------------------------------------------------------------------
# Speaker Verification — ECAPA-TDNN (ONNX Runtime)
# ---------------------------------------------------------------------------
SPEAKER_MODEL_PATH = os.getenv("SPEAKER_MODEL_PATH", "models/ecapa_tdnn.onnx")
MASTER_EMBEDDING_PATH = os.getenv(
    "MASTER_EMBEDDING_PATH", "models/master_embedding.npy"
)

# Cosine similarity threshold — 0.0 (no match) to 1.0 (perfect match).
# Typical operational sweet-spot: 0.45–0.60
SPEAKER_VERIFY_THRESHOLD = float(
    os.getenv("SPEAKER_VERIFY_THRESHOLD", "0.50")
)

# Minimum audio duration (seconds) required for reliable embedding
SPEAKER_MIN_AUDIO_S = float(os.getenv("SPEAKER_MIN_AUDIO_S", "1.0"))

# ---------------------------------------------------------------------------
# Vision — Pi Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "320"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "240"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "15"))

# ---------------------------------------------------------------------------
# Face Tracking — MediaPipe BlazeFace
# ---------------------------------------------------------------------------
FACE_MIN_DETECTION_CONFIDENCE = float(
    os.getenv("FACE_MIN_DETECTION_CONFIDENCE", "0.55")
)
FACE_MODEL_SELECTION = int(os.getenv("FACE_MODEL_SELECTION", "0"))  # 0=short, 1=full

# Number of consecutive frames with no detection before declaring "lost"
FACE_LOST_FRAME_THRESHOLD = int(os.getenv("FACE_LOST_FRAME_THRESHOLD", "15"))

# ---------------------------------------------------------------------------
# Visual Servoing — PID Controller Gains
# ---------------------------------------------------------------------------
# Proportional, Integral, Derivative gains for head Yaw (horizontal) axis
SERVO_YAW_KP = float(os.getenv("SERVO_YAW_KP", "0.25"))
SERVO_YAW_KI = float(os.getenv("SERVO_YAW_KI", "0.01"))
SERVO_YAW_KD = float(os.getenv("SERVO_YAW_KD", "0.05"))

# PID gains for head Pitch (vertical) axis
SERVO_PITCH_KP = float(os.getenv("SERVO_PITCH_KP", "0.20"))
SERVO_PITCH_KI = float(os.getenv("SERVO_PITCH_KI", "0.01"))
SERVO_PITCH_KD = float(os.getenv("SERVO_PITCH_KD", "0.04"))

# Deadzone in normalized coordinates — below this error, don't move
SERVO_DEADZONE = float(os.getenv("SERVO_DEADZONE", "0.05"))

# Max angular velocity sent per servo tick (radians)
SERVO_MAX_SPEED = float(os.getenv("SERVO_MAX_SPEED", "0.15"))

# Head Yaw angle (radians) beyond which the body rotates to realign
BODY_REALIGN_YAW_THRESHOLD_RAD = float(
    os.getenv("BODY_REALIGN_YAW_THRESHOLD_RAD", "0.52")  # ~30 degrees
)

# Body turn speed when realigning (rad/s)
BODY_TURN_SPEED = float(os.getenv("BODY_TURN_SPEED", "0.3"))

# Servo loop target interval (seconds)
SERVO_LOOP_INTERVAL_S = float(os.getenv("SERVO_LOOP_INTERVAL_S", "0.066"))  # ~15 Hz

# NAO Head joint limits (radians) — safety clamps
NAO_HEAD_YAW_MIN = -2.0857
NAO_HEAD_YAW_MAX = 2.0857
NAO_HEAD_PITCH_MIN = -0.6720
NAO_HEAD_PITCH_MAX = 0.5149

# ---------------------------------------------------------------------------
# Object Detection — YOLOv8-Nano (TFLite)
# ---------------------------------------------------------------------------
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/yolov8n.tflite")
YOLO_CONFIDENCE_THRESHOLD = float(
    os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.40")
)
YOLO_INPUT_SIZE = (320, 320)  # Width x Height for the TFLite input tensor
YOLO_MAX_SEARCH_SECONDS = float(os.getenv("YOLO_MAX_SEARCH_SECONDS", "30.0"))

# COCO class names subset most relevant to personal assistant
YOLO_TARGET_CLASSES = {
    "keys": ["cell phone", "remote"],  # proxy COCO classes for small objects
    "phone": ["cell phone"],
    "bottle": ["bottle"],
    "cup": ["cup"],
    "person": ["person"],
    "book": ["book"],
    "bag": ["backpack", "handbag", "suitcase"],
}

# ---------------------------------------------------------------------------
# Memory Management
# ---------------------------------------------------------------------------
RAM_WARNING_THRESHOLD_MB = int(os.getenv("RAM_WARNING_THRESHOLD_MB", "200"))
RAM_CRITICAL_THRESHOLD_MB = int(os.getenv("RAM_CRITICAL_THRESHOLD_MB", "100"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(name)-14s] %(levelname)-7s %(message)s"
