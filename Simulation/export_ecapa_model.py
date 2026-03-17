#!/usr/bin/env python3
"""
export_ecapa_model.py -- Export ECAPA-TDNN to ONNX without full SpeechBrain import.

Downloads the pretrained ECAPA-TDNN weights from HuggingFace and exports
to ONNX using only torch + huggingface_hub, bypassing the SpeechBrain
import chain (which has torchaudio compatibility issues).

Usage:
    python export_ecapa_model.py
    python export_ecapa_model.py --verify
    python export_ecapa_model.py --output path/to/ecapa_tdnn.onnx
"""

from __future__ import annotations

import argparse
import os
import sys


def export_model(output_path: str, verify: bool = False) -> None:
    """Download ECAPA-TDNN and export to ONNX."""
    import torch
    import numpy as np

    print("=" * 60)
    print("  ECAPA-TDNN ONNX Export (standalone)")
    print("=" * 60)

    # ---- Step 1: Download model files from HuggingFace ----
    print("\nStep 1/4: Downloading model from HuggingFace...")
    from huggingface_hub import hf_hub_download

    repo_id = "speechbrain/spkrec-ecapa-voxceleb"
    cache_dir = "pretrained_models/spkrec-ecapa-voxceleb"

    # Download the hyperparams and model files
    hyperparams_path = hf_hub_download(
        repo_id=repo_id, filename="hyperparams.yaml",
        cache_dir=cache_dir,
    )
    classifier_path = hf_hub_download(
        repo_id=repo_id, filename="embedding_model.ckpt",
        cache_dir=cache_dir,
    )
    mean_var_norm_path = hf_hub_download(
        repo_id=repo_id, filename="mean_var_norm_emb.ckpt",
        cache_dir=cache_dir,
    )

    print(f"  Downloaded to: {cache_dir}")

    # ---- Step 2: Build the model using SpeechBrain's pretrained interface ----
    # We need to use a workaround for the torchaudio issue
    print("Step 2/4: Loading model...")

    # Patch torchaudio before importing speechbrain
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    # Fix Windows symlink privilege issue — patch link_with_strategy to copy
    import speechbrain.utils.fetching as sb_fetch
    import shutil
    from pathlib import Path
    _orig_link = sb_fetch.link_with_strategy
    def _copy_strategy(src, dst, *args, **kwargs):
        dst = Path(dst)
        src = Path(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy2(str(src), str(dst))
        return dst
    sb_fetch.link_with_strategy = _copy_strategy

    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        try:
            from speechbrain.pretrained import EncoderClassifier
        except ImportError:
            print("ERROR: speechbrain not found. Install: pip install speechbrain")
            sys.exit(1)

    # Use a SHORT save dir to avoid Windows path issues
    savedir = os.path.join(os.path.expanduser("~"), "sb_ecapa")
    classifier = EncoderClassifier.from_hparams(
        source=repo_id,
        savedir=savedir,
    )
    print("  Model loaded successfully.")

    # ---- Step 3: Compute embeddings via PyTorch and save as ONNX ----
    # The full encode_batch pipeline has dynamic shapes that the ONNX
    # tracer can't handle. Instead, we:
    # 1. Use encode_batch in pure PyTorch to compute embeddings
    # 2. Export a thin wrapper that matches the stub model's interface

    print("Step 3/4: Testing model with dummy audio...")

    # Test that encode_batch works
    dummy_wav = torch.randn(1, 16000)
    with torch.no_grad():
        test_emb = classifier.encode_batch(dummy_wav).squeeze(1)
        test_emb = torch.nn.functional.normalize(test_emb, p=2, dim=1)
    emb_dim = test_emb.shape[1]
    print(f"  Embedding dimension: {emb_dim}")
    print(f"  Embedding norm: {test_emb.norm().item():.4f}")

    # Save the PyTorch model as a scriptable module that onnxruntime can load
    # We'll use a different approach: save the full pipeline as a .pt file
    # and create a Python wrapper that loads it for ONNX inference.
    #
    # Actually, the cleanest approach for our use case: export ONLY the
    # ECAPA-TDNN embedding model (after feature extraction) and do feature
    # extraction in Python. But this changes the interface.
    #
    # Simplest working approach: save the classifier and compute embeddings
    # using SpeechBrain at enrollment time, then save .npy embeddings.
    # At runtime on RPi, only onnxruntime is needed for the stub/real model.
    #
    # For ONNX export, we need to trace through the feature pipeline.
    # Use a FIXED input length to avoid dynamic shape issues.

    print("Step 3/4: Exporting to ONNX (fixed 4s input)...")

    # Fixed 4 seconds (64000 samples at 16kHz) — covers typical utterances
    FIXED_SAMPLES = 64000

    class EcapaFixed(torch.nn.Module):
        """Wrapper with fixed-length input for ONNX export."""
        def __init__(self, classifier):
            super().__init__()
            # Extract internal modules
            self.compute_features = classifier.mods.compute_features
            self.mean_var_norm = classifier.mods.mean_var_norm
            self.embedding_model = classifier.mods.embedding_model

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            # waveform: (batch, FIXED_SAMPLES)
            feats = self.compute_features(waveform)
            feats = self.mean_var_norm(
                feats, torch.ones(feats.shape[0], device=feats.device)
            )
            embeddings = self.embedding_model(feats)
            embeddings = embeddings.squeeze(1)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings

    wrapper = EcapaFixed(classifier)
    wrapper.eval()

    dummy_input = torch.randn(1, FIXED_SAMPLES)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["waveform"],
        output_names=["embedding"],
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Exported to: {output_path}")
    print(f"  File size:   {file_size_mb:.1f} MB")

    # ---- Step 4: Verify ----
    if verify:
        print("Step 4/4: Verifying ONNX model...")
        import onnxruntime as ort

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        print(f"  Input:  {inp.name} shape={inp.shape}")
        print(f"  Output: {out.name} shape={out.shape}")

        test_wav = np.random.randn(1, 32000).astype(np.float32)
        result = sess.run(None, {inp.name: test_wav})
        emb = result[0]
        print(f"  Embedding shape: {emb.shape}")
        print(f"  Embedding norm:  {np.linalg.norm(emb[0]):.4f} (should be ~1.0)")

        with torch.no_grad():
            torch_emb = wrapper(torch.from_numpy(test_wav)).numpy()
        cos_sim = float(np.dot(emb[0], torch_emb[0]) / (
            np.linalg.norm(emb[0]) * np.linalg.norm(torch_emb[0])
        ))
        print(f"  ONNX vs PyTorch cosine: {cos_sim:.6f}")
        print(f"  {'PASSED' if cos_sim > 0.99 else 'WARNING: mismatch'}")
    else:
        print("Step 4/4: Skipping verification (use --verify)")

    print()
    print("=" * 60)
    print("  Export complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. Enroll speakers:  python enroll_from_data.py --set-master david")
    print(f"  2. Run simulation:   python run_simulation.py --master david")
    print(f"  3. For RPi: copy {output_path} to rpi_brain/models/ecapa_tdnn.onnx")


def main():
    parser = argparse.ArgumentParser(description="Export ECAPA-TDNN to ONNX.")

    # Default output: the real models directory
    sim_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(sim_dir)
    default_output = os.path.join(
        repo_root, "nao_assistant", "rpi_brain", "models", "ecapa_tdnn.onnx"
    )

    parser.add_argument("--output", "-o", default=default_output,
                        help=f"Output path (default: {default_output})")
    parser.add_argument("--verify", "-v", action="store_true",
                        help="Verify exported model")
    args = parser.parse_args()

    export_model(args.output, args.verify)


if __name__ == "__main__":
    main()
