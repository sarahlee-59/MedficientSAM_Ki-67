"""ONNX vs OpenVINO speed benchmark — 5 bench images, real click prompts.

구조:
  - 이미지당 encode 1회 측정
  - 셀(4개)별 decode 개별 측정
  - e2e = encode + 4 decode 합산
  - 워밍업 1회 후 REPEAT회 평균
"""
import sys
import time
import json
from pathlib import Path

import cv2
import numpy as np

# ── 경로 설정 ──────────────────────────────────────────────────────────────
BASE       = Path("/mnt/Disk1/sylee")
IMAGE_DIR  = BASE / "Ki-67_service/frontend/public/samples"
JSON_DIR   = BASE / "Ki-67_service/benchmark/results"
ONNX_DIR   = BASE / "Ki-67_service/deployment"
OV_DIR     = BASE / "Ki-67_service/deployment/openvino"

sys.path.insert(0, str(ONNX_DIR / "onnx"))
import infer as onnx_infer

import importlib.util
spec = importlib.util.spec_from_file_location("ov_infer", OV_DIR / "infer.py")
ov_infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ov_infer)

# ── JSON → 이미지별 셀 포인트 로드 ───────────────────────────────────────
# { "bench1.png": [ np.array shape (K,2), ... ], ... }
CELLS: dict[str, list[np.ndarray]] = {}
for jf in sorted(JSON_DIR.glob("*.json")):
    d = json.loads(jf.read_text())
    img_name = d["image"]
    pts_list = []
    for cell in d["cells"]:
        xy = np.array([[p["x"], p["y"]] for p in cell["prompt_points"]], dtype=np.float32)
        pts_list.append(xy)
    CELLS[img_name] = pts_list

REPEAT = 5

# ── 모델 로드 ──────────────────────────────────────────────────────────────
print("모델 로딩 중...")
onnx_seg = onnx_infer.Ki67Segmenter(
    encoder_path=ONNX_DIR / "encoder.quantized.onnx",
    decoder_path=ONNX_DIR / "decoder.quantized.onnx",
)
ov_seg = ov_infer.Ki67Segmenter(
    encoder_path=OV_DIR / "encoder.xml",
    decoder_path=OV_DIR / "decoder.xml",
)
print("로딩 완료\n")


def bench_image(seg, img_rgb: np.ndarray, cells_pts: list[np.ndarray], label: str):
    """encode 1회 + 셀별 decode 측정. 워밍업 후 REPEAT회 평균."""
    H, W = img_rgb.shape[:2]

    # 워밍업
    emb = seg.encode(img_rgb)
    for pts in cells_pts:
        seg.decode(emb, pts[np.newaxis], (H, W))

    # encode 측정
    enc_times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        emb = seg.encode(img_rgb)
        enc_times.append(time.perf_counter() - t0)
    enc_ms = np.mean(enc_times) * 1000

    # decode 측정 (셀별)
    dec_times = []
    for pts in cells_pts:
        times = []
        for _ in range(REPEAT):
            t0 = time.perf_counter()
            seg.decode(emb, pts[np.newaxis], (H, W))
            times.append(time.perf_counter() - t0)
        dec_times.append(np.mean(times) * 1000)

    dec_total_ms = sum(dec_times)
    e2e_ms = enc_ms + dec_total_ms

    dec_str = " + ".join(f"{t:.1f}" for t in dec_times)
    print(f"  [{label:10s}] encode: {enc_ms:6.1f} ms  |  decode(×{len(cells_pts)}): {dec_str} = {dec_total_ms:.1f} ms  |  e2e: {e2e_ms:.1f} ms")
    return enc_ms, dec_total_ms, e2e_ms


# ── 실행 ───────────────────────────────────────────────────────────────────
onnx_enc, onnx_dec, onnx_e2e = [], [], []
ov_enc,   ov_dec,   ov_e2e   = [], [], []

for img_name, cells_pts in CELLS.items():
    img_path = IMAGE_DIR / img_name
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print(f"\n{img_name}  ({len(cells_pts)} cells)")
    e, d, t = bench_image(onnx_seg, img_rgb, cells_pts, "ONNX-INT8")
    onnx_enc.append(e); onnx_dec.append(d); onnx_e2e.append(t)

    e, d, t = bench_image(ov_seg, img_rgb, cells_pts, "OV-FP32")
    ov_enc.append(e); ov_dec.append(d); ov_e2e.append(t)

# ── 요약 ───────────────────────────────────────────────────────────────────
def avg(lst): return np.mean(lst)

print("\n" + "=" * 62)
print(f"{'':22s}  {'ONNX-INT8':>10}  {'OV-FP32':>10}  {'OV/ONNX':>8}")
print("-" * 62)
for label, oa, va in [
    ("encode 평균 (ms)",  avg(onnx_enc), avg(ov_enc)),
    ("decode 평균 (ms)",  avg(onnx_dec), avg(ov_dec)),
    ("e2e 평균 (ms)",     avg(onnx_e2e), avg(ov_e2e)),
]:
    ratio = va / oa
    faster = "OV 빠름" if ratio < 1 else "ONNX 빠름"
    print(f"{label:22s}  {oa:>10.1f}  {va:>10.1f}  {ratio:>6.2f}x  {faster}")
print("=" * 62)
