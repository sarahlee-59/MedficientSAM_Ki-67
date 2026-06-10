"""Ki-67 segmentation inference server (OpenVINO FP32 backend).

Run from this directory:
    uvicorn server:app --host 0.0.0.0 --port 8001

Requires: fastapi uvicorn[standard] python-multipart openvino numpy opencv-python
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from infer import Ki67Segmenter

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

_segmenter: Ki67Segmenter | None = None
_embedding_cache: dict = {}


def get_segmenter() -> Ki67Segmenter:
    global _segmenter
    if _segmenter is None:
        base = Path(__file__).parent
        _segmenter = Ki67Segmenter(
            encoder_path=base / "encoder.xml",
            decoder_path=base / "decoder.xml",
        )
    return _segmenter


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    points: str = Form(...),
    labels: str = Form(...),
):
    img_bytes = await image.read()
    img_hash = hashlib.md5(img_bytes).hexdigest()

    seg = get_segmenter()

    if img_hash not in _embedding_cache:
        img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        embedding = seg.encode(img_rgb)
        _embedding_cache.clear()
        _embedding_cache[img_hash] = (embedding, H, W)

    embedding, H, W = _embedding_cache[img_hash]

    pts = np.array(json.loads(points), dtype=np.float32)   # (K, 2)
    lbls = np.array(json.loads(labels), dtype=np.float32)  # (K,)

    masks = seg.decode(
        embedding,
        pts[np.newaxis],    # (1, K, 2)
        (H, W),
        lbls[np.newaxis],   # (1, K)
    )  # (1, H, W) uint8

    return {"mask": masks[0].flatten().tolist(), "width": W, "height": H}
