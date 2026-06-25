"""Ki-67 segmentation inference server (OpenVINO FP32 backend).

Run from this directory:
    uvicorn server:app --host 0.0.0.0 --port 8000

Requires: fastapi uvicorn[standard] python-multipart openvino numpy opencv-python
"""
import json
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from infer import Ki67Segmenter

app = FastAPI()


@app.on_event("startup")
async def startup():
    get_segmenter()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """미처리 예외도 JSON으로 반환 (프론트 프록시가 res.json()으로 파싱하므로 필수)."""
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


def _decode_image(img_bytes: bytes) -> np.ndarray:
    img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_array is None:
        raise HTTPException(status_code=400, detail="이미지를 디코딩할 수 없습니다. 파일이 손상되었거나 지원하지 않는 형식입니다.")
    return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

_segmenter: Ki67Segmenter | None = None
_session_cache: dict = {}   # session_id → (embedding, H, W)


def get_segmenter() -> Ki67Segmenter:
    global _segmenter
    if _segmenter is None:
        base = Path(__file__).parent
        _segmenter = Ki67Segmenter(
            encoder_path=base / "encoder.xml",
            decoder_path=base / "decoder.xml",
        )
    return _segmenter


def _decode_session(session_id: str, points_json: str, labels_json: str) -> dict:
    if session_id not in _session_cache:
        raise HTTPException(status_code=404, detail="session not found")
    embedding, H, W = _session_cache[session_id]
    pts = np.array(json.loads(points_json), dtype=np.float32)
    lbls = np.array(json.loads(labels_json), dtype=np.float32)
    masks = get_segmenter().decode(
        embedding, pts[np.newaxis], (H, W), lbls[np.newaxis]
    )
    return {"mask": masks[0].flatten().tolist(), "width": W, "height": H}


@app.post("/encode")
async def encode(image: UploadFile = File(...)):
    """이미지 인코딩 후 session_id 반환. 이후 /decode로 좌표만 전송 가능."""
    img_bytes = await image.read()
    img_rgb = _decode_image(img_bytes)
    H, W = img_rgb.shape[:2]
    embedding = get_segmenter().encode(img_rgb)
    session_id = str(uuid.uuid4())
    _session_cache.clear()
    _session_cache[session_id] = (embedding, H, W)
    return {"session_id": session_id, "width": W, "height": H}


class DecodeRequest(BaseModel):
    session_id: str
    points: list
    labels: list


@app.post("/decode")
async def decode(req: DecodeRequest):
    """session_id + 좌표만으로 마스크 반환 (이미지 전송 없음)."""
    return _decode_session(
        req.session_id,
        json.dumps(req.points),
        json.dumps(req.labels),
    )


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    points: str = Form(...),
    labels: str = Form(...),
):
    """단일 엔드포인트 fallback (encode+decode 통합)."""
    img_bytes = await image.read()
    img_rgb = _decode_image(img_bytes)
    H, W = img_rgb.shape[:2]
    embedding = get_segmenter().encode(img_rgb)
    session_id = str(uuid.uuid4())
    _session_cache.clear()
    _session_cache[session_id] = (embedding, H, W)
    return _decode_session(session_id, points, labels)
