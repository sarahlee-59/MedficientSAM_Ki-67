"""ki67_hybrid_bench1_negative.json의 cell 1~4(negative, 3/4/5/4 prompt)를
OpenVINO(polyline) 결과와 Torch(best.ckpt) 추론 결과 각각 실제 GT 인스턴스와 겹쳐서 시각화.
venv_medficientsam python으로 실행 (best.ckpt 추론에 torch/hydra 필요).
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import albumentations.augmentations.geometric.transforms as _alb_geo_transforms
import albumentations.augmentations.geometric.flip as _alb_geo_flip
for _name in dir(_alb_geo_flip):
    if not hasattr(_alb_geo_transforms, _name):
        setattr(_alb_geo_transforms, _name, getattr(_alb_geo_flip, _name))

import rootutils

REPO = Path("/mnt/Disk1/sylee/DP_MedificientSAM")
rootutils.setup_root(str(REPO), indicator=".project-root", pythonpath=True)

import hydra
from hydra import compose, initialize_config_dir

JSON_PATH = Path(__file__).parent / "ki67_hybrid_bench1_negative.json"
IMAGE_PATH = Path("/mnt/Disk1/sylee/frontend/public/samples/bench1.png")
GT_PATH = Path(
    "/mnt/Disk1/sylee/train_npz/Pathology_new/gts_npz_s128/2D_S26-02873,A3,FDC00099,37255_3810_1_78609_72394_802_807_r00_c00.npz"
)
OUT_DIR = Path(__file__).parent / "final"
OUT_DIR.mkdir(exist_ok=True)

# polyline(OpenVINO 추론 결과)과 GT 인스턴스의 최대 겹침으로 매칭된 cell -> GT id
CELL_TO_GT_ID = {1: 36, 2: 56, 3: 27, 4: 55}
PAD = 15
SCALE = 4

IMAGE_ENCODER_INPUT_SIZE = 512
PROMPT_ENCODER_INPUT_SIZE = 512


def overlay_and_save(image_bgr, gt_mask, pred_mask, color, out_path):
    H, W = gt_mask.shape
    ys, xs = np.where((gt_mask > 0) | (pred_mask > 0))
    x0, x1 = max(0, xs.min() - PAD), min(W, xs.max() + PAD)
    y0, y1 = max(0, ys.min() - PAD), min(H, ys.max() + PAD)

    crop = image_bgr[y0:y1, x0:x1]
    gt_crop = gt_mask[y0:y1, x0:x1]
    pred_crop = pred_mask[y0:y1, x0:x1]

    inter = int((gt_crop & pred_crop).sum())
    union = int((gt_crop | pred_crop).sum())
    iou = inter / union if union else float("nan")
    dice = 2 * inter / (gt_crop.sum() + pred_crop.sum()) if (gt_crop.sum() + pred_crop.sum()) else float("nan")

    out = crop.copy().astype(np.float32) * 0.55
    gt_layer = np.zeros_like(out); gt_layer[gt_crop > 0] = (0, 0, 255)
    pred_layer = np.zeros_like(out); pred_layer[pred_crop > 0] = color
    out += gt_layer * 0.45 + pred_layer * 0.45
    out = np.clip(out, 0, 255).astype(np.uint8)

    gt_cnt, _ = cv2.findContours(gt_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pred_cnt, _ = cv2.findContours(pred_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, gt_cnt, -1, (0, 0, 255), 1)
    cv2.drawContours(out, pred_cnt, -1, color, 1)

    out_big = cv2.resize(out, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_path), out_big)
    return iou, dice


def run_openvino(data, image_bgr, gt):
    H, W = image_bgr.shape[:2]
    for cell in data["cells"]:
        idx = cell["cell_index"]
        gt_id = CELL_TO_GT_ID[idx]
        n_prompts = len(cell["prompt_points"])

        poly = np.array([[p["x"], p["y"]] for p in cell["polyline"]], dtype=np.int32)
        pred_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(pred_mask, [poly], 1)
        gt_mask = (gt == gt_id).astype(np.uint8)

        out_path = OUT_DIR / f"cell{idx}_p{n_prompts}_vs_real_gt.png"
        iou, dice = overlay_and_save(image_bgr, gt_mask, pred_mask, (0, 255, 0), out_path)
        print(f"[OpenVINO] cell{idx} (GT id={gt_id}): IoU={iou:.4f}, Dice={dice:.4f} -> {out_path}")


def preprocess(image_rgb_u8: np.ndarray):
    H, W = image_rgb_u8.shape[:2]
    scale = IMAGE_ENCODER_INPUT_SIZE / max(H, W)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))
    resized = cv2.resize(image_rgb_u8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32)
    mn, mx = float(resized.min()), float(resized.max())
    scaled = (resized - mn) / max(mx - mn, 1e-8)
    padded = np.zeros((IMAGE_ENCODER_INPUT_SIZE, IMAGE_ENCODER_INPUT_SIZE, 3), dtype=np.float32)
    padded[:new_h, :new_w, :] = scaled
    return padded, new_h, new_w


def run_torch(data, image_bgr, image_rgb, gt):
    H, W = image_rgb.shape[:2]

    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="infer.yaml",
            overrides=[
                "experiment=infer_finetuned_l1",
                "device=cpu",
                "+model.finetune_lit_module.strict=False",
            ],
        )
    model = hydra.utils.instantiate(cfg.model).to("cpu")
    model.eval()

    pre, new_h, new_w = preprocess(image_rgb)
    img_t = torch.from_numpy(pre).permute(2, 0, 1).unsqueeze(0).float()
    ratio = PROMPT_ENCODER_INPUT_SIZE / max(H, W)

    with torch.no_grad():
        # warmup
        warm_emb = model.image_encoder(img_t)
        wpts = np.array([[p["x"], p["y"]] for p in data["cells"][0]["prompt_points"]], dtype=np.float32) * ratio
        wc = torch.from_numpy(wpts).unsqueeze(0).float()
        wl = torch.ones((1, wpts.shape[0]), dtype=torch.float32)
        se, de = model.prompt_encoder(points=(wc, wl), boxes=None, masks=None)
        model.mask_decoder(
            image_embeddings=warm_emb, image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
            multimask_output=model.multimask_output,
        )

        image_embedding = model.image_encoder(img_t)

        for cell in data["cells"]:
            idx = cell["cell_index"]
            gt_id = CELL_TO_GT_ID[idx]

            pts = np.array([[p["x"], p["y"]] for p in cell["prompt_points"]], dtype=np.float32)
            n_prompts = len(pts)
            coords = torch.from_numpy(pts * ratio).unsqueeze(0).float()
            labels = torch.ones((1, pts.shape[0]), dtype=torch.float32)

            sparse_emb, dense_emb = model.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=model.multimask_output,
            )
            mask_out = model.postprocess_masks(low_res_masks, (new_h, new_w), (H, W))
            pred_mask = (mask_out.squeeze().cpu().numpy() > 0).astype(np.uint8)
            gt_mask = (gt == gt_id).astype(np.uint8)

            # pred_mask에 GT와 멀리 떨어진 의사양성(stray) 조각이 섞여 있으면 crop bbox가
            # 불필요하게 늘어지므로, GT와 가장 많이 겹치는 연결요소만 시각화에 사용.
            n_comp, comp = cv2.connectedComponents(pred_mask)
            if n_comp > 2:
                best_comp, best_overlap = None, -1
                for c in range(1, n_comp):
                    overlap = int(((comp == c) & (gt_mask > 0)).sum())
                    if overlap > best_overlap:
                        best_overlap, best_comp = overlap, c
                crop_pred_mask = (comp == best_comp).astype(np.uint8)
                print(f"  [info] cell{idx}: GT와 떨어진 분리 조각 {n_comp - 2}개 존재 (시각화는 주 영역만, IoU/Dice는 전체 마스크 기준)")
            else:
                crop_pred_mask = pred_mask

            out_path = OUT_DIR / f"cell{idx}_p{n_prompts}_torch_vs_real_gt.png"
            iou, dice = overlay_and_save(image_bgr, gt_mask, crop_pred_mask, (255, 0, 0), out_path)
            print(f"[Torch] cell{idx} (n_points={n_prompts}, GT id={gt_id}): IoU={iou:.4f}, Dice={dice:.4f} -> {out_path}")


def main():
    data = json.loads(JSON_PATH.read_text())
    image_bgr = cv2.imread(str(IMAGE_PATH))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gt = np.load(GT_PATH)["gts"]

    run_openvino(data, image_bgr, gt)
    run_torch(data, image_bgr, image_rgb, gt)


if __name__ == "__main__":
    main()
