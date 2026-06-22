"""ki67_hybrid_bench1 (4).json의 cell 1~4 prompt_points만 원본 이미지 위에 표시.
원본 이미지는 밝기를 보정해서 더 밝게 표현."""
import json
from pathlib import Path

import cv2
import numpy as np

JSON_PATH = Path('/mnt/Disk1/sylee/benchmark/gt_comparison/cells/ki67_hybrid_bench1_positive.json')
IMAGE_PATH = Path(
    "/mnt/Disk1/sylee/benchmark/speed/images/"
    "2D_S26-02873,A3,FDC00099,37255_3810_1_78609_72394_802_807_r00_c00.png"
)
OUT_PATH = Path('/mnt/Disk1/sylee/benchmark/gt_comparison/overlays/bench1_mask_overlay.png')

COLORS = {1: (0, 255, 0), 2: (0, 255, 255), 3: (255, 0, 255), 4: (255, 128, 0)}  # BGR
BRIGHTNESS_GAMMA = 0.6  # <1이면 더 밝게


def brighten(image_bgr: np.ndarray, gamma: float = BRIGHTNESS_GAMMA) -> np.ndarray:
    norm = image_bgr.astype(np.float32) / 255.0
    out = np.power(norm, gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    data = json.loads(JSON_PATH.read_text())
    image_bgr = cv2.imread(str(IMAGE_PATH))

    overlay = brighten(image_bgr)

    for cell in data["cells"]:
        idx = cell["cell_index"]
        color = COLORS[idx]

        poly = np.array([[p["x"], p["y"]] for p in cell["polyline"]], dtype=np.int32)
        cx, cy = poly[:, 0].mean(), poly[:, 1].mean()

        for p in cell["prompt_points"]:
            cv2.circle(overlay, (int(p["x"]), int(p["y"])), 2, color, -1)
            cv2.circle(overlay, (int(p["x"]), int(p["y"])), 2, (0, 0, 0), 1)

        cv2.putText(overlay, f"cell{idx}", (int(cx) - 10, int(cy) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    cv2.imwrite(str(OUT_PATH), overlay)
    print(f"saved -> {OUT_PATH}  (마스크 없이 prompt point만 표시)")


if __name__ == "__main__":
    main()
