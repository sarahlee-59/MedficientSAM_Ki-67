# E11_holdout INT8 ONNX deployment package

Ki67 nucleus instance segmentation, point-prompt only. Distilled from MedSAM
(efficientvit-l1 image encoder + MedSAM vit_b prompt/mask decoder), fine-tuned
on combined nucleus dataset (Ki67 + PanNuke + MoNuSeg via challenge Google
Sheet, 818 Ki67 tiles held out), decoder-only training, then exported to ONNX and INT8 dynamic-quantized.

## Files (copy all into a single directory)

| File | Size | Purpose |
|---|---|---|
| `encoder.quantized.onnx` | ~44 MB | INT8 image encoder (input: HxWx3 uint8 RGB) |
| `decoder.quantized.onnx` | ~9 MB | INT8 mask decoder (point prompts only) |
| `infer.py` | — | Inference module (`Ki67Segmenter` class) |
| `server.py` | — | FastAPI HTTP inference server |
| `example.py` | — | Minimal CLI usage demo |
| `README.md` | — | This file |

Total model size: ~53 MB.

## Runtime requirements

```
python >= 3.10
onnxruntime >= 1.16
numpy
opencv-python
fastapi          # server.py only
uvicorn          # server.py only
python-multipart # server.py only
Pillow           # only required by example.py
```

```bash
pip install onnxruntime numpy opencv-python fastapi "uvicorn[standard]" python-multipart
```

No PyTorch, no torchvision, no training-codebase dependencies.

## HTTP Server

`server.py`는 FastAPI 기반 추론 서버입니다. Next.js 프론트엔드와 연동하거나 HTTP로 직접 호출할 때 사용합니다.

```bash
cd deployment
uvicorn server:app --host 0.0.0.0 --port 8000
```

### API

**`POST /infer`** (multipart/form-data)

| 필드 | 타입 | 설명 |
|---|---|---|
| `image` | file | PNG/JPEG 이미지 |
| `points` | string (JSON) | `[[x1,y1], [x2,y2], ...]` 픽셀 좌표 |
| `labels` | string (JSON) | `[1, 1, 0, ...]` (1=전경, 0=배경) |

**Response**

```json
{
  "mask": [0, 1, 0, ...],
  "width": 256,
  "height": 256
}
```

`mask`는 `(H × W)` 크기의 uint8 flat 배열 (0 또는 1)입니다.

서버는 이미지 MD5 해시 기준으로 인코더 임베딩을 캐시합니다. 같은 이미지에서 반복 클릭 시 인코더를 한 번만 실행합니다.

### curl 예제

```bash
curl -X POST http://localhost:8000/infer \
  -F "image=@my_tile.png" \
  -F 'points=[[120,80],[130,95]]' \
  -F 'labels=[1,1]'
```

---

## Python 직접 사용

```python
import numpy as np
from infer import Ki67Segmenter

seg = Ki67Segmenter(
    encoder_path="encoder.quantized.onnx",
    decoder_path="decoder.quantized.onnx",
)

# image: (H, W, 3) uint8 RGB tile.
# points: (N, K, 2) float32 — N instances × K positive clicks each, (x, y) in tile coords.
image = ...
points = np.array([[[120.0, 80.0]],
                   [[200.0, 150.0]]])
masks = seg.predict(image, points)  # (N, H, W) uint8 binary masks
```

Multiple clicks per instance:
```python
points = np.array([
    [[100., 80.], [130., 95.], [115., 110.]],
    [[200., 150.], [220., 165.], [210., 175.]],
])
masks = seg.predict(image, points)
```

Mixing positive and negative clicks:
```python
points = np.array([[[120., 80.], [50., 50.]]])
labels = np.array([[1., 0.]], dtype=np.float32)  # 1=fg, 0=bg, -1=padding
masks = seg.predict(image, points, point_labels=labels)
```

If you'll feed many prompt sets per image, encode once:
```python
emb = seg.encode(image)
masks_a = seg.decode(emb, points_a, image.shape[:2])
masks_b = seg.decode(emb, points_b, image.shape[:2])
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
- **Box prompts are not supported** by this decoder.

## Quick sanity check

```bash
python example.py my_tile.png --points "120,80"
python example.py my_tile.png --points "100,80;130,95;115,110"
python example.py my_tile.png --points "100,80;130,95;115,110 | 200,150;220,165;210,175"
python example.py my_tile.png --points "120,80;130,95" --neg-points "50,50;60,60"
```

Coordinates are in tile pixels (x = column, y = row), origin top-left.
Results saved as `<image>.overlay.png`.
