"""E11_holdout (Ki67 nucleus segmentation) ONNX inference, point prompt only.

Self-contained — needs only onnxruntime + numpy + opencv-python.
Drop this file next to encoder.quantized.onnx + decoder.quantized.onnx and use:

    from infer import Ki67Segmenter
    seg = Ki67Segmenter(
        encoder_path="encoder.quantized.onnx",
        decoder_path="decoder.quantized.onnx",
    )
    # image: (H, W, 3) uint8 RGB; points: (N, K, 2) float32 (x, y) in tile coords
    masks = seg.predict(image, points)   # (N, H, W) uint8 binary masks

Model: E11_holdout (combined nucleus − 818 hold-out, MedSAM vit_b prompt/mask
decoder + efficientvit-l1 distill image encoder, decoder-only fine-tune,
INT8 dynamic-quantized weights). Validated on n=100 Ki67 hold-out tiles:
INT8 dice within FP32 ± SE for k=1/3/5 point prompts.

Tile resolution: trained on 128–256 px tiles. The encoder resizes the
longest side to 512 internally. Larger tiles work but may degrade.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort

IMAGE_ENCODER_INPUT_SIZE = 512
PROMPT_ENCODER_INPUT_SIZE = 1024


class Ki67Segmenter:
    """Two-stage ONNX inference: encoder → (cache) → decoder per prompt set."""

    def __init__(
        self,
        encoder_path: str | Path,
        decoder_path: str | Path,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ):
        self.encoder = ort.InferenceSession(str(encoder_path), providers=list(providers))
        self.decoder = ort.InferenceSession(str(decoder_path), providers=list(providers))
        # Sanity: encoder.onnx must be the preprocess_image=true variant
        enc_inputs = {i.name for i in self.encoder.get_inputs()}
        if "original_size" not in enc_inputs:
            raise ValueError(
                f"encoder.onnx is missing 'original_size' input — this loader is for "
                f"the in-model preprocess variant (onnx-e11_holdout). Inputs: {sorted(enc_inputs)}"
            )

    def encode(self, image: np.ndarray) -> np.ndarray:
        """image: (H, W, 3) uint8 RGB → image_embeddings (1, 256, 64, 64) float32."""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3) uint8, got {image.shape} {image.dtype}")
        H, W = image.shape[:2]
        return self.encoder.run(
            ["image_embeddings"],
            {"image": image, "original_size": np.array([H, W], dtype=np.int16)},
        )[0]

    def decode(
        self,
        image_embeddings: np.ndarray,
        points_tile: np.ndarray,
        original_size: tuple[int, int],
        point_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Decoder pass for N instances × K clicks.

        image_embeddings : (1, 256, 64, 64) float32 — output of self.encode()
        points_tile      : (N, K, 2) float32 in tile pixel coords (x, y)
        original_size    : (H, W) of the source tile, needed to scale points
        point_labels     : (N, K) float32, 1=positive, 0=negative, -1=padding.
                           Default = all-positive ones.

        Returns: (N, H, W) uint8 binary masks at tile resolution.
        """
        if points_tile.ndim != 3 or points_tile.shape[-1] != 2:
            raise ValueError(f"points_tile must be (N, K, 2), got {points_tile.shape}")
        N, K, _ = points_tile.shape
        H, W = original_size

        # Scale tile coords → prompt_encoder space (1024). The encoder uses
        # resize-longest-side, so the same ratio applies to point coords.
        ratio = PROMPT_ENCODER_INPUT_SIZE / max(H, W)
        points_pe = (points_tile * ratio).astype(np.float32)

        if point_labels is None:
            point_labels = np.ones((N, K), dtype=np.float32)
        point_labels = point_labels.astype(np.float32)

        # Decoder returns (N, 1, 512, 512) float32 mask logits.
        masks_512 = self.decoder.run(
            ["masks"],
            {
                "image_embeddings": image_embeddings.astype(np.float32),
                "point_coords": points_pe,
                "point_labels": point_labels,
            },
        )[0]

        # The encoder resize-longest-sides the image to fit in
        # (image_encoder_input_size, image_encoder_input_size) and pads the
        # right/bottom with zeros, so the (512, 512) decoder mask is only
        # *valid* over the (new_h, new_w) top-left region. Must crop before
        # the final upsample — otherwise the padded zone gets stretched into
        # the output and the mask appears squished toward one side.
        new_h = int(round(H * IMAGE_ENCODER_INPUT_SIZE / max(H, W)))
        new_w = int(round(W * IMAGE_ENCODER_INPUT_SIZE / max(H, W)))

        out = np.empty((N, H, W), dtype=np.uint8)
        for i in range(N):
            valid = masks_512[i, 0, :new_h, :new_w]
            resized = cv2.resize(valid, (W, H), interpolation=cv2.INTER_LINEAR)
            out[i] = (resized > 0).astype(np.uint8)
        return out

    def predict(
        self,
        image: np.ndarray,
        points_tile: np.ndarray,
        point_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """End-to-end: encode + decode in one call."""
        H, W = image.shape[:2]
        emb = self.encode(image)
        return self.decode(emb, points_tile, (H, W), point_labels=point_labels)
