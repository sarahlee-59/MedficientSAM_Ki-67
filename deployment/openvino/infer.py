"""E11_holdout (Ki67 nucleus segmentation) OpenVINO inference, point prompt only.

Self-contained — needs only openvino + numpy + opencv-python.
Drop this file next to encoder.xml/.bin + decoder.xml/.bin and use:

    from infer import Ki67Segmenter
    seg = Ki67Segmenter(
        encoder_path="encoder.xml",
        decoder_path="decoder.xml",
    )
    # image: (H, W, 3) uint8 RGB; points: (N, K, 2) float32 (x, y) in tile coords
    masks = seg.predict(image, points)   # (N, H, W) uint8 binary masks

API surface is **identical** to the INT8 ONNX package's `Ki67Segmenter`
(deployment/e11_holdout_int8/infer.py): constructor takes `encoder_path` /
`decoder_path` (just point at `.xml` instead of `.onnx`), and `encode` /
`decode` / `predict` signatures match exactly. Callers can swap backends
by changing only the import line and the file paths.

Model: E11_holdout (combined nucleus − 818 hold-out, MedSAM vit_b prompt/mask
decoder + efficientvit-l1 distill image encoder, decoder-only fine-tune,
FP32 OpenVINO IR converted from the cpp-variant ONNX export with
preprocess_image=false). Numerically verified vs PyTorch (binary-mask IoU
>= 0.9994 on synthetic prompts; see tools/verify_export_e11_holdout.py).

Performance: ~1.65× faster e2e than FP32 ONNX, ~1.9× faster than INT8 ONNX
on Intel CPU — Intel oneDNN fused kernels via OV CPU plugin.

Tile resolution: trained on 128–256 px tiles. The encoder resizes the
longest side to 512 internally. Larger tiles work but may degrade.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import openvino as ov

IMAGE_ENCODER_INPUT_SIZE = 512
PROMPT_ENCODER_INPUT_SIZE = 1024


class Ki67Segmenter:
    """Two-stage OpenVINO inference: encoder → (cache) → decoder per prompt set.

    Unlike the ONNX in-model preprocess variant, this graph expects a
    pre-resized/scaled/padded (512, 512, 3) float32 tensor as encoder input;
    the resize-longest-side + min-max scale + zero-pad is done here on the host
    (see `preprocess()`).
    """

    def __init__(
        self,
        encoder_path: str | Path,
        decoder_path: str | Path,
        device: str = "CPU",
    ):
        """encoder_path / decoder_path : path to the OpenVINO .xml file. The
        matching .bin must sit next to it (same stem) — OpenVINO loads it
        implicitly. Argument names mirror the INT8 ONNX package so callers can
        swap backends by changing only the import line and the file paths."""
        core = ov.Core()
        if device == "CPU":
            # Force FP32 inference precision + latency-oriented scheduling — single
            # interactive request, not throughput batching.
            core.set_property("CPU", ov.properties.hint.inference_precision(ov.Type.f32))
            core.set_property("CPU", ov.properties.hint.performance_mode(
                ov.properties.hint.PerformanceMode.LATENCY))
        self.encoder = core.compile_model(str(encoder_path), device)
        self.decoder = core.compile_model(str(decoder_path), device)
        self._enc_out = self.encoder.output(0)
        self._dec_out = self.decoder.output(0)

    # ----- preprocess (host-side) -----

    @staticmethod
    def preprocess(image_rgb_u8: np.ndarray) -> np.ndarray:
        """Resize longest side → 512, min-max scale to [0, 1], pad right/bottom
        to (512, 512, 3) float32. Mirrors the EncoderOnnxModel with
        preprocess_image=false + scale_image=True."""
        if image_rgb_u8.dtype != np.uint8 or image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
            raise ValueError(
                f"image must be (H, W, 3) uint8 RGB, got {image_rgb_u8.shape} {image_rgb_u8.dtype}"
            )
        H, W = image_rgb_u8.shape[:2]
        scale = IMAGE_ENCODER_INPUT_SIZE / max(H, W)
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))
        resized = cv2.resize(image_rgb_u8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized = resized.astype(np.float32)
        mn, mx = float(resized.min()), float(resized.max())
        scaled = (resized - mn) / max(mx - mn, 1e-8)
        padded = np.zeros(
            (IMAGE_ENCODER_INPUT_SIZE, IMAGE_ENCODER_INPUT_SIZE, 3), dtype=np.float32
        )
        padded[:new_h, :new_w, :] = scaled
        return padded

    # ----- encode / decode -----

    def encode(self, image_rgb_u8: np.ndarray) -> np.ndarray:
        """image: (H, W, 3) uint8 RGB → image_embeddings (1, 256, 64, 64) float32.

        Includes the host-side preprocess. Cache this once per image and reuse
        across many prompt sets via `decode()` — encoder dominates wall time."""
        pre = self.preprocess(image_rgb_u8)
        return self.encoder({"image": pre})[self._enc_out]

    def decode_logits(
        self,
        image_embeddings: np.ndarray,
        points_tile: np.ndarray,
        original_size: tuple[int, int],
        point_labels: Optional[np.ndarray] = None,
        blur_sigma: float = 0.8,
    ) -> np.ndarray:
        """Decoder pass returning the raw mask **logit** field (no threshold).

        Same arguments as `decode()`. Returns: (N, H, W) float32 mask logits at
        tile resolution. The 0-level iso-contour (logit == 0, i.e. prob == 0.5)
        is the cell boundary; keeping the continuous field lets the client trace
        a sub-pixel-accurate contour (marching squares) instead of a staircased
        one from a binarised mask.

        blur_sigma : light Gaussian (in tile px) applied to the logit field before
        return. Small cells make 1-2 px logit noise show up as needle spikes on
        the iso-contour; smoothing the continuous field suppresses them at the
        source while preserving sub-pixel boundary position. 0 disables.
        """
        if points_tile.ndim != 3 or points_tile.shape[-1] != 2:
            raise ValueError(f"points_tile must be (N, K, 2), got {points_tile.shape}")
        N, K, _ = points_tile.shape
        H, W = original_size

        # Original-pixel coord → prompt_encoder space (1024). Same resize ratio
        # as the encoder, since the encoder did longest-side-fits-512.
        ratio = PROMPT_ENCODER_INPUT_SIZE / max(H, W)
        points_pe = (points_tile * ratio).astype(np.float32)

        if point_labels is None:
            point_labels = np.ones((N, K), dtype=np.float32)
        point_labels = point_labels.astype(np.float32)

        # Decoder returns (N, 1, 512, 512) float32 mask logits.
        masks_512 = self.decoder({
            "image_embeddings": image_embeddings.astype(np.float32),
            "point_coords": points_pe,
            "point_labels": point_labels,
        })[self._dec_out]

        # The encoder padded the bottom/right of the 512x512 input — only the
        # (new_h, new_w) top-left region of the decoder output is valid. Crop
        # before upsampling, otherwise the padded zone gets stretched.
        new_h = int(round(H * IMAGE_ENCODER_INPUT_SIZE / max(H, W)))
        new_w = int(round(W * IMAGE_ENCODER_INPUT_SIZE / max(H, W)))

        out = np.empty((N, H, W), dtype=np.float32)
        for i in range(N):
            valid = masks_512[i, 0, :new_h, :new_w]
            resized = cv2.resize(valid, (W, H), interpolation=cv2.INTER_LINEAR)
            if blur_sigma > 0:
                resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=blur_sigma)
            out[i] = resized
        return out

    def decode(
        self,
        image_embeddings: np.ndarray,
        points_tile: np.ndarray,
        original_size: tuple[int, int],
        point_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Decoder pass for N instances × K clicks.

        image_embeddings : (1, 256, 64, 64) float32 — output of `self.encode()`.
        points_tile      : (N, K, 2) float32 in tile pixel coords (x, y).
        original_size    : (H, W) of the source tile, needed to scale points
                           into the prompt encoder's 1024-pixel space.
        point_labels     : (N, K) float32, 1=positive, 0=negative, -1=padding.
                           Default = all-positive ones.

        Returns: (N, H, W) uint8 binary masks at tile resolution.
        """
        return (self.decode_logits(
            image_embeddings, points_tile, original_size, point_labels
        ) > 0).astype(np.uint8)

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
