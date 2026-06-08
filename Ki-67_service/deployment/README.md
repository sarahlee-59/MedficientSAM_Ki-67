# E11_holdout INT8 ONNX deployment package

Ki67 nucleus instance segmentation, point-prompt only. Distilled from MedSAM
(efficientvit-l1 image encoder + MedSAM vit_b prompt/mask decoder), fine-tuned
on combined nucleus dataset (Ki67 + PanNuke + MoNuSeg, 818 Ki67 tiles held
out), decoder-only training, then exported to ONNX and INT8 dynamic-quantized.

## Files (copy all into a single directory)

| File | Size | Purpose |
|---|---|---|
| `encoder.quantized.onnx` | ~44 MB | INT8 image encoder (input: HxWx3 uint8 RGB) |
| `decoder.quantized.onnx` | ~9 MB | INT8 mask decoder (point prompts only) |
| `infer.py` | — | Inference module (`Ki67Segmenter` class) |
| `example.py` | — | Minimal usage demo |
| `README.md` | — | This file |

Total model size: ~53 MB.

## Runtime requirements

```
python >= 3.10
onnxruntime >= 1.16
numpy
opencv-python
Pillow            # only required by example.py for image I/O — infer.py itself
                  # does not depend on it
```

`pip install onnxruntime numpy opencv-python Pillow` is enough — no PyTorch,
no torchvision, no training-codebase dependencies. Drop `Pillow` if you're
only using `Ki67Segmenter` from your own code and providing the image as a
numpy array.

CPU inference is the validated path. GPU works with the appropriate
onnxruntime build (`onnxruntime-gpu`) and `providers=("CUDAExecutionProvider",)`.

## Usage

```python
import numpy as np
from infer import Ki67Segmenter

seg = Ki67Segmenter(
    encoder_path="encoder.quantized.onnx",
    decoder_path="decoder.quantized.onnx",
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

If you'll feed many prompt sets per image, encode once:
```python
emb = seg.encode(image)                                                  # one encoder pass
masks_a = seg.decode(emb, points_a, image.shape[:2])                     # cheap decode
masks_b = seg.decode(emb, points_b, image.shape[:2])                     # cheap decode
```

## Performance notes

- **Tile size**: trained on 128–256 px tiles. Larger inputs are resized to a
  longest-side of 512 internally; very large tiles will lose fine detail.
- **Prompt convention**: positive clicks should fall *inside* the target
  nucleus. For `k=3`/`k=5` clicks per instance, regular-polygon vertices
  inside the mask (≈10 px radius from centroid) produce the best dice.
- **Validated dice** (Ki67 hold-out 818 tiles, n=100, seed=42):
  - k=1 click: ~0.30
  - k=3 clicks: ~0.73
  - k=5 clicks: ~0.76
- **Box prompts are not supported** by this decoder. If you need bounding-box
  inference, request the box-capable export separately.

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
