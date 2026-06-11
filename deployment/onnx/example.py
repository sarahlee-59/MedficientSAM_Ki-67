"""Minimal usage example for Ki67Segmenter — image in, mask overlay out.

This is what a deployment user actually needs: load any RGB image, give a few
click coordinates, get the predicted masks and a visualization PNG.

Usage:
    python example.py image.jpg --points "120,80;200,150"
        → saves image.overlay.png next to the input

Multiple clicks per cell (separated by ';' within an instance, '|' between):
    python example.py image.jpg --points "100,80;130,95;115,110 | 200,150;220,165;210,175"
        → 2 instances × 3 clicks each (regular triangle style)

Background clicks (label=0) for refinement: use --neg-points "x,y;..." per instance,
in the same | -separated layout as --points.
"""
from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from infer import Ki67Segmenter

HERE = Path(__file__).resolve().parent
ENCODER = HERE / "encoder.quantized.onnx"
DECODER = HERE / "decoder.quantized.onnx"


def parse_point_groups(spec: str) -> list[list[tuple[float, float]]]:
    """'x,y;x,y | x,y;x,y' → [[(x,y),(x,y)], [(x,y),(x,y)]]. None on empty."""
    if not spec:
        return []
    groups = []
    for inst in spec.split("|"):
        pts = []
        for pair in inst.strip().split(";"):
            pair = pair.strip()
            if not pair:
                continue
            x_s, y_s = pair.split(",")
            pts.append((float(x_s), float(y_s)))
        if pts:
            groups.append(pts)
    return groups


def draw_overlay(
    image: np.ndarray,
    masks: np.ndarray,
    fg_points: list[list[tuple[float, float]]],
    bg_points: list[list[tuple[float, float]]],
) -> np.ndarray:
    """Alpha-blend masks per instance + draw clicks. Returns RGB uint8."""
    out = image.copy()
    for i, m in enumerate(masks):
        # distinct color per instance (golden-ratio hue rotation)
        h = (i * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        color = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.float32)

        idx = m > 0
        out[idx] = (out[idx].astype(np.float32) * 0.6 + color * 0.4).astype(np.uint8)
        contours, _ = cv2.findContours(
            (m > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, tuple(int(c) for c in color), 2)

    # Foreground clicks: green dots
    for inst_pts in fg_points:
        for x, y in inst_pts:
            cv2.circle(out, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.circle(out, (int(x), int(y)), 6, (0, 0, 0), 1)
    # Background clicks: blue dots
    for inst_pts in bg_points:
        for x, y in inst_pts:
            cv2.circle(out, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.circle(out, (int(x), int(y)), 6, (0, 0, 0), 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path, help="RGB image path (jpg/png/tif)")
    p.add_argument("--points", required=True,
                   help="Foreground clicks. 'x,y;x,y | x,y;x,y' (within '|' = same instance)")
    p.add_argument("--neg-points", default="",
                   help="Optional background clicks per instance (same layout as --points)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output overlay PNG path. Default: <image>.overlay.png next to input")
    args = p.parse_args()

    if not args.image.exists():
        sys.exit(f"image not found: {args.image}")

    fg_groups = parse_point_groups(args.points)
    bg_groups = parse_point_groups(args.neg_points)
    if not fg_groups:
        sys.exit("at least one --points click is required")
    if bg_groups and len(bg_groups) != len(fg_groups):
        sys.exit(
            f"--neg-points must have same instance count as --points "
            f"({len(bg_groups)} vs {len(fg_groups)})"
        )

    # Pad to a regular (N, K, 2) tensor — use label=-1 for padded slots so
    # the model ignores them.
    K = max(len(g) + (len(bg_groups[i]) if bg_groups else 0) for i, g in enumerate(fg_groups))
    N = len(fg_groups)
    pts = np.zeros((N, K, 2), dtype=np.float32)
    lbls = -np.ones((N, K), dtype=np.float32)
    for i, fg in enumerate(fg_groups):
        bg = bg_groups[i] if bg_groups else []
        all_pts = fg + bg
        all_lbls = [1.0] * len(fg) + [0.0] * len(bg)
        for j, ((x, y), lab) in enumerate(zip(all_pts, all_lbls)):
            pts[i, j] = (x, y)
            lbls[i, j] = lab

    image = np.array(Image.open(args.image).convert("RGB"))
    print(f"loaded image {image.shape} {image.dtype}, {N} instance(s), max K={K}")

    seg = Ki67Segmenter(encoder_path=ENCODER, decoder_path=DECODER)
    masks = seg.predict(image, pts, point_labels=lbls)

    for i, m in enumerate(masks):
        n_fg = len(fg_groups[i])
        n_bg = len(bg_groups[i]) if bg_groups else 0
        print(f"  inst {i}: clicks fg={n_fg} bg={n_bg}  predicted_fg_pixels={int(m.sum()):,}")

    out_path = args.out or args.image.with_suffix(args.image.suffix + ".overlay.png")
    overlay = draw_overlay(image, masks, fg_groups, bg_groups)
    Image.fromarray(overlay).save(out_path)
    print(f"saved overlay → {out_path}")


if __name__ == "__main__":
    main()
