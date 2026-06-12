# E11_holdout FP32 OpenVINO deployment package

Ki67 nucleus instance segmentation, point-prompt only. Distilled from MedSAM
(efficientvit-l1 image encoder + MedSAM vit_b prompt/mask decoder), fine-tuned
on combined nucleus dataset (Ki67 + PanNuke + MoNuSeg via challenge Google
Sheet, 818 Ki67 tiles held out), decoder-only training, then exported to ONNX and converted to OpenVINO
FP32 IR.

This package is the **latency-oriented** deployment. For the smaller (~53 MB)
INT8 ONNX variant with equivalent accuracy, see the sibling `e11_holdout_int8/`
package — it's ~3.5× smaller on disk but ~1.65× slower e2e on Intel CPU.

## Files (copy all into a single directory)

| File | Size | Purpose |
|---|---|---|
| `encoder.xml` | 639 KB | Image encoder graph (OpenVINO IR) |
| `encoder.bin` | 167 MB | Image encoder FP32 weights |
| `decoder.xml` | 323 KB | Mask decoder graph (point prompts only) |
| `decoder.bin` | 19 MB | Mask decoder FP32 weights |
| `infer.py` | — | Inference module (`Ki67Segmenter` class) |
| `example.py` | — | Minimal CLI usage demo |
| `requirements.txt` | — | Pinned runtime dependencies |
| `README.md` | — | This file |

Total model size: ~187 MB. `.xml` and `.bin` must live side-by-side with the
same stem — OpenVINO loads the `.bin` implicitly when you pass the `.xml`.

## Runtime requirements

```
python >= 3.10
openvino >= 2024.0
numpy
opencv-python
Pillow            # only required by example.py for image I/O — infer.py itself
                  # does not depend on it
```

`pip install -r requirements.txt` is enough — no PyTorch, no onnxruntime, no
training-codebase dependencies.

CPU inference is the validated path (Intel oneDNN fused kernels). Other
devices: pass `device="GPU"` etc. to `Ki67Segmenter(...)` if you have the
appropriate OpenVINO plugin installed; otherwise leave as default.

## Usage

```python
import numpy as np
from infer import Ki67Segmenter

seg = Ki67Segmenter(
    encoder_path="encoder.xml",
    decoder_path="decoder.xml",
)

# image: (H, W, 3) uint8 RGB tile.
# points: (N, K, 2) float32 — N instances × K positive clicks each, (x, y) in tile coords.
image = ...   # H&E or IHC RGB tile, any HxW
points = np.array([[[120.0, 80.0]],         # instance 0: one click at (x=120, y=80)
                   [[200.0, 150.0]]])       # instance 1: one click
masks = seg.predict(image, points)          # (N, H, W) uint8 binary masks
```

Multiple clicks per instance:
```python
# K=3 positive clicks per instance, e.g., regular polygon vertices.
points = np.array([
    [[100., 80.], [130., 95.], [115., 110.]],   # instance 0
    [[200., 150.], [220., 165.], [210., 175.]], # instance 1
])
masks = seg.predict(image, points)
```

Mixing positive and negative clicks:
```python
points = np.array([[[120., 80.], [50., 50.]]])      # 2 clicks for 1 instance
labels = np.array([[1., 0.]], dtype=np.float32)     # 1=fg, 0=bg, -1=padding
masks = seg.predict(image, points, point_labels=labels)
```

If you'll feed many prompt sets per image (e.g. an interactive web demo —
click → re-segment), encode once and reuse the embedding:
```python
emb = seg.encode(image)                                  # one encoder pass (~140 ms)
masks_a = seg.decode(emb, points_a, image.shape[:2])     # cheap decode (~10 ms)
masks_b = seg.decode(emb, points_b, image.shape[:2])     # cheap decode
```

The API is **identical** to the INT8 ONNX package's `Ki67Segmenter`, so
switching backends only requires changing the file paths:
```python
# Before (INT8 ONNX)
seg = Ki67Segmenter(encoder_path="encoder.quantized.onnx",
                    decoder_path="decoder.quantized.onnx")
# After (FP32 OV)
seg = Ki67Segmenter(encoder_path="encoder.xml",
                    decoder_path="decoder.xml")
```
The `Ki67Segmenter` class is named the same in both packages but they live
in separate directories with separate `infer.py` files — keep them apart
on `PYTHONPATH` (typical: install one or the other, not both).

## Performance notes

Measured on Intel CPU, 807×802 Ki67 tile, single inference request, median of
30 trials after 5 warmups (`tools/bench_onnx_vs_openvino.py`):

| Backend | encoder | decoder N=1 K=1 | e2e | vs FP32 ONNX |
|---|---|---|---|---|
| FP32 ONNX (onnxruntime) | 236 ms | 11.0 ms | 247 ms | baseline |
| INT8 ONNX (onnxruntime) | 274 ms | 10.0 ms | 284 ms | 0.87× (slower!) |
| **FP32 OV (this package)** | **139 ms** | **9.2 ms** | **148 ms** | **1.67×** |

For interactive use (e.g. "encode once, click N times to refine"), the
encoder runs once (~140 ms) and each additional click pays only the decoder
cost (~10 ms per instance) — feels real-time.

## Model details

- **Tile size**: trained on 128–256 px tiles. Larger inputs are resized to a
  longest-side of 512 internally; very large tiles will lose fine detail.
- **Prompt convention**: positive clicks should fall *inside* the target
  nucleus. For `k=3`/`k=5` clicks per instance, regular-polygon vertices
  inside the mask (≈10 px radius from centroid) produce the best dice.
- **Validated dice** (Ki67 hold-out 818 tiles, n=100, seed=42, FP32):
  - k=1 click: ~0.30
  - k=3 clicks: ~0.73
  - k=5 clicks: ~0.76
- **Box prompts are not supported** by this decoder. If you need bounding-box
  inference, request the box-capable export separately.

## Numerical equivalence

This IR was verified against the PyTorch reference (binary-mask IoU
≥ 0.9994 on synthetic prompts across N=1/2 × K=1/3 cases — only encoder
embedding max_abs diff ~5e-6, decoder logit max_abs ~3e-2 but threshold>0
agreement essentially perfect). See `tools/verify_export_e11_holdout.py`
in the training repository for the equivalence harness.

## Quick sanity check

Run `example.py` on any RGB image with explicit click coordinates and save an
overlay PNG:

```bash
# Single instance, single click:
python example.py my_tile.png --points "120,80"

# Single instance, 3 positive clicks (regular-triangle style):
python example.py my_tile.png --points "100,80;130,95;115,110"

# Two instances, 3 clicks each — '|' separates instances, ';' separates clicks:
python example.py my_tile.png --points "100,80;130,95;115,110 | 200,150;220,165;210,175"

# Mix foreground + background clicks (background = label 0 for refinement):
python example.py my_tile.png --points "120,80;130,95" --neg-points "50,50;60,60"
```

Coordinates are in tile pixels (x = column, y = row), with origin top-left.
The script prints per-instance fg pixel counts and saves `<image>.overlay.png`
showing predicted masks (distinct color per instance, with click dots: green
for foreground, blue for background).
