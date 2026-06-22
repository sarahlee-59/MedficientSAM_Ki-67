"""bench1.png == pathology npz 원본 이미지. negative.json의 cell 1~4 polyline 결과를
각각 매칭된 실제 GT 인스턴스와 겹쳐서 시각화.
"""
import json
from pathlib import Path

import cv2
import numpy as np

JSON_PATH = Path('/mnt/Disk1/sylee/benchmark/gt_comparison/cells/ki67_hybrid_bench1_negative.json')
IMAGE_PATH = Path('/mnt/Disk1/sylee/frontend/public/samples/bench1.png')
GT_PATH = Path(
    "/mnt/Disk1/sylee/train_npz/Pathology_new/gts_npz_s128/2D_S26-02873,A3,FDC00099,37255_3810_1_78609_72394_802_807_r00_c00.npz"
)
OUT_DIR = Path('/mnt/Disk1/sylee/benchmark/gt_comparison/overlays/negative')
OUT_DIR.mkdir(exist_ok=True)

CELL_TO_GT_ID = {1: 36, 2: 27, 3: 38, 4: 56}
PAD = 15
SCALE = 4

data = json.loads(JSON_PATH.read_text())
image = cv2.imread(str(IMAGE_PATH))
H, W = image.shape[:2]
gt = np.load(GT_PATH)["gts"]

for cell in data["cells"]:
    idx = cell["cell_index"]
    gt_id = CELL_TO_GT_ID[idx]

    poly = np.array([[p["x"], p["y"]] for p in cell["polyline"]], dtype=np.int32)
    pred_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(pred_mask, [poly], 1)
    gt_mask = (gt == gt_id).astype(np.uint8)

    ys, xs = np.where((gt_mask > 0) | (pred_mask > 0))
    x0, x1 = max(0, xs.min() - PAD), min(W, xs.max() + PAD)
    y0, y1 = max(0, ys.min() - PAD), min(H, ys.max() + PAD)

    crop = image[y0:y1, x0:x1]
    gt_crop = gt_mask[y0:y1, x0:x1]
    pred_crop = pred_mask[y0:y1, x0:x1]

    inter = int((gt_crop & pred_crop).sum())
    union = int((gt_crop | pred_crop).sum())
    iou = inter / union if union else float("nan")
    dice = 2 * inter / (gt_crop.sum() + pred_crop.sum())

    out = crop.copy().astype(np.float32) * 0.55
    gt_layer = np.zeros_like(out); gt_layer[gt_crop > 0] = (0, 0, 255)
    pred_layer = np.zeros_like(out); pred_layer[pred_crop > 0] = (0, 255, 0)
    out += gt_layer * 0.45 + pred_layer * 0.45
    out = np.clip(out, 0, 255).astype(np.uint8)

    gt_cnt, _ = cv2.findContours(gt_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pred_cnt, _ = cv2.findContours(pred_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, gt_cnt, -1, (0, 0, 255), 1)
    cv2.drawContours(out, pred_cnt, -1, (0, 255, 0), 1)

    out_big = cv2.resize(out, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    out_path = OUT_DIR / f"cell{idx}_vs_real_gt.png"
    cv2.imwrite(str(out_path), out_big)

    print(f"cell{idx} (GT id={gt_id}): GT area={gt_crop.sum()}, pred area={pred_crop.sum()}, IoU={iou:.4f}, Dice={dice:.4f} -> {out_path}")
