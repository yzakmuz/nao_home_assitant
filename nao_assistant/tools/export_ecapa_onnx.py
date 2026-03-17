#!/usr/bin/env python3
"""
export_ecapa_onnx.py -- Export SpeechBrain ECAPA-TDNN to ONNX format.

This script downloads the pretrained ECAPA-TDNN model from HuggingFace
(speechbrain/spkrec-ecapa-voxceleb) and exports it to an ONNX file that
can be loaded by onnxruntime on the Raspberry Pi for speaker verification.

Requirements (install on a PC, NOT the RPi):
    pip install speechbrain torch onnx onnxruntime numpy

Usage:
    python export_ecapa_onnx.py
    python export_ecapa_onnx.py --output ../rpi_brain/models/ecapa_tdnn.onnx
    python export_ecapa_onnx.py --verify  # verify exported model works

The exported model:
    - Input:  "waveform" — float32 tensor of shape (batch, samples), 16 kHz mono
    - Output: "embedding" — float32 tensor of shape (batch, 192)
    - Size:   ~25 MB (vs 328 KB stub)

After exporting, copy the .onnx file to the RPi at:
    rpi_brain/models/ecapa_tdnn.onnx
Then re-run enrollment:
    cd rpi_brain && python enroll_speaker.py
"""

from __future__ import annotations

import argparse
import os
import sys


def export_model(output_path: str, verify: bool = False) -> None:
    """Download ECAPA-TDNN from SpeechBrain and export to ONNX."""

    print("=" * 60)
    print("  ECAPA-TDNN ONNX Export")
    print("=" * 60)

    # ---- Step 1: Import dependencies ----
    try:
        import torch
        import numpy as np
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Install with: pip install torch numpy")
        sys.exit(1)

    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        # SpeechBrain >= 1.0 moved the class
        try:
            from speechbrain.pretrained import EncoderClassifier
        except ImportError:
            print("\nSpeechBrain not found. Install with: pip install speechbrain")
            sys.exit(1)

    # ---- Step 2: Load pretrained model ----
    print("\nStep 1/4: Downloading ECAPA-TDNN from HuggingFace...")
    print("  Source: speechbrain/spkrec-ecapa-voxceleb")
    print("  (This may take a minute on first run)")

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
    )

    # ---- Step 3: Create ONNX-compatible wrapper ----
    print("Step 2/4: Creating ONNX wrapper...")

    class EcapaOnnxWrapper(torch.nn.Module):
        """Wrapper that takes raw waveform and returns L2-normalized embedding."""

        def __init__(self, encoder, mean_var_norm, embedding_model):
            super().__init__()
            self.encoder = encoder
            self.mean_var_norm = mean_var_norm
            self.embedding_model = embedding_model

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            """
            Args:
                waveform: (batch, samples) float32 at 16 kHz

            Returns:
                embedding: (batch, 192) L2-normalized float32
            """
            # Compute features
            feats = self.encoder.mods.compute_features(waveform)
            feats = self.encoder.mods.mean_var_norm(feats, torch.ones(feats.shape[0]))
            embeddings = self.embedding_model(feats)
            # embeddings shape: (batch, 1, 192)
            embeddings = embeddings.squeeze(1)
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings

    # Try to extract the components
    # SpeechBrain's EncoderClassifier has:
    #   .mods.compute_features  (Fbank)
    #   .mods.mean_var_norm     (InputNormalization)
    #   .mods.embedding_model   (ECAPA_TDNN)
    try:
        wrapper = EcapaOnnxWrapper(
            encoder=classifier,
            mean_var_norm=classifier.mods.mean_var_norm,
            embedding_model=classifier.mods.embedding_model,
        )
    except AttributeError:
        print("\nCould not access SpeechBrain model internals.")
        print("Falling back to direct encode_batch export...")
        # Fallback: wrap encode_batch directly
        class SimpleWrapper(torch.nn.Module):
            def __init__(self, classifier):
                super().__init__()
                self.classifier = classifier

            def forward(self, waveform):
                embeddings = self.classifier.encode_batch(waveform)
                embeddings = embeddings.squeeze(1)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                return embeddings

        wrapper = SimpleWrapper(classifier)

    wrapper.eval()

    # ---- Step 4: Export to ONNX ----
    print("Step 3/4: Exporting to ONNX...")

    # Create a dummy input (1 second of audio at 16kHz)
    dummy_input = torch.randn(1, 16000)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["waveform"],
        output_names=["embedding"],
        dynamic_axes={
            "waveform": {0: "batch", 1: "samples"},
            "embedding": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Exported to: {output_path}")
    print(f"  File size:   {file_size_mb:.1f} MB")

    # ---- Step 5: Verify (optional) ----
    if verify:
        print("Step 4/4: Verifying exported model...")
        import onnxruntime as ort

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        print(f"  Input:  {inp.name} shape={inp.shape} type={inp.type}")
        print(f"  Output: {out.name} shape={out.shape} type={out.type}")

        # Test with random audio
        test_wav = np.random.randn(1, 32000).astype(np.float32)
        result = sess.run(None, {inp.name: test_wav})
        emb = result[0]
        print(f"  Test embedding shape: {emb.shape}")
        print(f"  Test embedding norm:  {np.linalg.norm(emb[0]):.4f} (should be ~1.0)")

        # Compare with PyTorch output
        with torch.no_grad():
            torch_emb = wrapper(torch.from_numpy(test_wav)).numpy()
        cos_sim = np.dot(emb[0], torch_emb[0]) / (
            np.linalg.norm(emb[0]) * np.linalg.norm(torch_emb[0])
        )
        print(f"  ONNX vs PyTorch cosine similarity: {cos_sim:.6f} (should be ~1.0)")

        if cos_sim > 0.99:
            print("  VERIFICATION PASSED")
        else:
            print("  WARNING: ONNX output differs from PyTorch — check export")
    else:
        print("Step 4/4: Skipping verification (use --verify to enable)")

    print()
    print("=" * 60)
    print("  Export complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. Copy {output_path} to the RPi at rpi_brain/models/ecapa_tdnn.onnx")
    print("  2. Run enrollment: cd rpi_brain && python enroll_speaker.py")
    print("  3. Test: python -c \"from audio.speaker_verify import SpeakerVerifier; SpeakerVerifier()\"")
    print()
    print("IMPORTANT: After replacing the model, you MUST re-enroll.")
    print("The old master_embedding.npy is incompatible with the new model.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SpeechBrain ECAPA-TDNN to ONNX for speaker verification."
    )
    parser.add_argument(
        "--output", "-o",
        default="ecapa_tdnn.onnx",
        help="Output path for the ONNX file (default: ecapa_tdnn.onnx)",
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Verify the exported ONNX model against PyTorch output",
    )
    args = parser.parse_args()

    export_model(args.output, verify=args.verify)


if __name__ == "__main__":
    main()
