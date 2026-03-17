"""
bootstrap.py -- Master import patcher for PC simulation.

MUST be called ONCE before importing anything from rpi_brain or nao_body.
Patches sys.modules and sys.path so that:
  - `from settings import ...` resolves to sim_config
  - `from naoqi import ALProxy` resolves to a fake stub
  - `import motion_library` works (nao_body on sys.path)
  - `vision.camera.Camera` is replaced by PcCamera
  - `vision.object_detector.ObjectDetector` is replaced by PcObjectDetector
"""

from __future__ import annotations

import os
import sys
import types

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(_SIM_DIR)
_RPI_BRAIN_DIR = os.path.join(_REPO_ROOT, "nao_assistant", "rpi_brain")
_NAO_BODY_DIR = os.path.join(_REPO_ROOT, "nao_assistant", "nao_body")


def bootstrap(sim_config_module) -> None:
    """Patch sys.modules and sys.path for PC simulation.

    Call this BEFORE importing anything from rpi_brain or nao_body.

    Args:
        sim_config_module: The sim_config module (already imported by caller).
    """
    # ------------------------------------------------------------------
    # Step 1: Install sim_config as "settings"
    # ------------------------------------------------------------------
    sys.modules["settings"] = sim_config_module

    # ------------------------------------------------------------------
    # Step 2: Add rpi_brain/ to sys.path so bare imports work
    #         (e.g., `from audio.mic_stream import MicStream`)
    # ------------------------------------------------------------------
    if _RPI_BRAIN_DIR not in sys.path:
        sys.path.insert(0, _RPI_BRAIN_DIR)

    # ------------------------------------------------------------------
    # Step 3: Fake naoqi module (server.py:36 does `from naoqi import ALProxy`)
    # ------------------------------------------------------------------
    fake_naoqi = types.ModuleType("naoqi")
    fake_naoqi.ALProxy = lambda name, ip, port: None  # type: ignore[attr-defined]
    sys.modules["naoqi"] = fake_naoqi

    # ------------------------------------------------------------------
    # Step 4: Add nao_body/ to sys.path (server.py:37 does `import motion_library`)
    # ------------------------------------------------------------------
    if _NAO_BODY_DIR not in sys.path:
        sys.path.insert(0, _NAO_BODY_DIR)

    # ------------------------------------------------------------------
    # Step 5: Add Simulation/ to sys.path (for our own adapter imports)
    # ------------------------------------------------------------------
    if _SIM_DIR not in sys.path:
        sys.path.insert(0, _SIM_DIR)

    # ------------------------------------------------------------------
    # Step 6: Monkey-patch Camera and ObjectDetector BEFORE main.py imports them
    # ------------------------------------------------------------------
    from vision import camera as cam_mod
    from adapters.pc_camera import PcCamera
    cam_mod.Camera = PcCamera  # type: ignore[misc]

    if not sim_config_module.SIM_NO_YOLO:
        from vision import object_detector as det_mod
        from adapters.pc_object_detector import PcObjectDetector
        det_mod.ObjectDetector = PcObjectDetector  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Step 6b: Monkey-patch PoseEstimator for fall detection GUI overlay
    # ------------------------------------------------------------------
    if getattr(sim_config_module, 'SIM_FALL_DETECTION', True):
        from vision import pose_estimator as pose_mod
        from adapters.pc_pose_estimator import PcPoseEstimator
        pose_mod.PoseEstimator = PcPoseEstimator  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Step 7: Patch audio modules if needed (no-mic / no-verify)
    #
    # CRITICAL: mic_stream.py has `import sounddevice as sd` at module level,
    # and stt_engine.py has `from vosk import ...` at module level.
    # If these packages are missing, we must install fake modules BEFORE
    # the real modules are imported by main.py.
    # ------------------------------------------------------------------
    if sim_config_module.SIM_NO_MIC:
        _ensure_audio_deps_available()
        _patch_audio_for_keyboard(sim_config_module)

    if sim_config_module.SIM_SKIP_VERIFY:
        _ensure_onnxruntime_available()
        _patch_speaker_verify()
    elif getattr(sim_config_module, 'SIM_USE_SPEECHBRAIN_VERIFY', False):
        _patch_speaker_verify_speechbrain()


def _ensure_audio_deps_available() -> None:
    """Install fake sounddevice/vosk if they're not installed.

    mic_stream.py does `import sounddevice as sd` at module level.
    stt_engine.py does `from vosk import KaldiRecognizer, Model, SetLogLevel`.
    If these are missing, we need stubs so the module imports don't crash
    (even though we replace the classes after import).
    """
    # Fake sounddevice
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        fake_sd = types.ModuleType("sounddevice")
        fake_sd.RawInputStream = type("RawInputStream", (), {  # type: ignore
            "__init__": lambda self, **kw: None,
            "start": lambda self: None,
            "stop": lambda self: None,
            "close": lambda self: None,
        })
        fake_sd.CallbackFlags = type("CallbackFlags", (), {})  # type: ignore
        sys.modules["sounddevice"] = fake_sd

    # Fake vosk
    try:
        import vosk  # noqa: F401
    except ImportError:
        fake_vosk = types.ModuleType("vosk")
        fake_vosk.Model = type("Model", (), {"__init__": lambda self, *a: None})  # type: ignore
        fake_vosk.KaldiRecognizer = type("KaldiRecognizer", (), {  # type: ignore
            "__init__": lambda self, *a, **kw: None,
        })
        fake_vosk.SetLogLevel = lambda level: None  # type: ignore
        sys.modules["vosk"] = fake_vosk


def _ensure_onnxruntime_available() -> None:
    """Install fake onnxruntime if not available."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.SessionOptions = type("SessionOptions", (), {  # type: ignore
            "__init__": lambda self: None,
            "inter_op_num_threads": 2,
            "intra_op_num_threads": 2,
            "graph_optimization_level": None,
        })
        fake_ort.GraphOptimizationLevel = type("GraphOptimizationLevel", (), {  # type: ignore
            "ORT_ENABLE_ALL": 0,
        })
        fake_ort.InferenceSession = type("InferenceSession", (), {  # type: ignore
            "__init__": lambda self, *a, **kw: None,
        })
        sys.modules["onnxruntime"] = fake_ort


def _patch_audio_for_keyboard(sim_config_module) -> None:
    """Replace MicStream and SttEngine with keyboard-based fallbacks."""
    from adapters.keyboard_input import DummyMicStream, KeyboardSttEngine
    from audio import mic_stream as mic_mod
    from audio import stt_engine as stt_mod

    mic_mod.MicStream = DummyMicStream  # type: ignore[misc]
    stt_mod.SttEngine = KeyboardSttEngine  # type: ignore[misc]


def _patch_speaker_verify_speechbrain() -> None:
    """Replace ONNX SpeakerVerifier with SpeechBrain-based PC verifier."""
    from adapters.pc_speaker_verify import PcSpeakerVerifier
    from audio import speaker_verify as sv_mod
    sv_mod.SpeakerVerifier = PcSpeakerVerifier  # type: ignore[misc]


def _patch_speaker_verify() -> None:
    """Make SpeakerVerifier always accept (skip verification)."""
    import numpy as np
    from audio import speaker_verify as sv_mod

    class AlwaysAcceptVerifier:
        """Stub verifier that always accepts."""
        def __init__(self) -> None:
            self._master_emb = None

        def compute_embedding(self, audio):
            return np.zeros(192, dtype=np.float32)

        def verify(self, audio):
            return (True, 1.0)

        def enroll(self, audio):
            return True

    sv_mod.SpeakerVerifier = AlwaysAcceptVerifier  # type: ignore[misc]
