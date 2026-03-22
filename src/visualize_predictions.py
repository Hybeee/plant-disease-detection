"""
visualize_predictions.py
------------------------
Renders side-by-side images: GT boxes (left) vs model predictions (right).

Usage
-----
# Single image
python visualize_predictions.py \
    --model  ssl_output/rounds/teacher_round_0.pt \
    --labels ssl_output/datasets/round_0/train/labels \
    --images dataset/images/train \
    --out    vis_round_0

# With original GT (before any merging)
python visualize_predictions.py \
    --model  ssl_output/rounds/teacher_round_2.pt \
    --labels dataset/labels/train \
    --images dataset/images/train \
    --out    vis_round_2_vs_original_gt

# Limit to N random images
python visualize_predictions.py ... --n 50

# Specific image
python visualize_predictions.py ... --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# Colour palette — one per class index, cycling if needed
_PALETTE = [
    (235,  64,  52), (235, 168,  52), ( 52, 235,  79), ( 52, 168, 235),
    (168,  52, 235), (235,  52, 168), ( 52, 235, 210), (210, 235,  52),
    (235, 122,  52), ( 52, 122, 235), (122, 235,  52), (235,  52, 122),
    ( 52, 235, 122), (122,  52, 235), (235, 210,  52), ( 52, 210, 235),
]

def _colour(cls_id: int) -> Tuple[int, int, int]:
    return _PALETTE[cls_id % len(_PALETTE)]


def _load_labels(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, int, int, int, int]]:
    """Read YOLO label file → list of (cls, x1, y1, x2, y2) in pixel coords."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = (float(x) for x in parts[1:5])
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def _draw_boxes(
    img: np.ndarray,
    boxes: List[Tuple[int, int, int, int, int]],
    class_names: List[str],
    conf_scores: Optional[List[float]] = None,
) -> np.ndarray:
    out = img.copy()
    for i, (cls_id, x1, y1, x2, y2) in enumerate(boxes):
        colour = _colour(cls_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        label = f"{name} {conf_scores[i]:.2f}" if conf_scores else name

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty), colour, -1)
        cv2.putText(out, label, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _make_side_by_side(
    img_path: Path,
    label_dir: Path,
    model: YOLO,
    class_names: List[str],
    conf_threshold: float,
) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    h, w = img.shape[:2]

    # GT side
    # Try to find the label by matching stem — handles the img_0000001 renaming
    # that _write_dataset_yaml does, as well as direct name matches.
    label_path = label_dir / f"{img_path.stem}.txt"
    gt_boxes   = _load_labels(label_path, w, h)
    left       = _draw_boxes(img, gt_boxes, class_names)

    # Prediction side
    result    = model.predict(str(img_path), conf=conf_threshold, verbose=False)[0]
    pred_boxes: List[Tuple[int, int, int, int, int]] = []
    confs: List[float] = []
    if result.boxes is not None and len(result.boxes):
        for box, conf, cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
        ):
            x1, y1, x2, y2 = (int(v) for v in box)
            pred_boxes.append((cls, x1, y1, x2, y2))
            confs.append(float(conf))
    right = _draw_boxes(img, pred_boxes, class_names, confs)

    # Header bars
    bar_h  = 32
    bar_gt = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar_pr = np.zeros((bar_h, w, 3), dtype=np.uint8)
    cv2.putText(bar_gt, f"GT  ({len(gt_boxes)} boxes)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(bar_pr, f"Pred  ({len(pred_boxes)} boxes)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

    left_col  = np.vstack([bar_gt, left])
    right_col = np.vstack([bar_pr, right])

    divider = np.full((left_col.shape[0], 3, 3), 60, dtype=np.uint8)
    return np.hstack([left_col, divider, right_col])


def _find_label_for_image(img_path: Path, label_dir: Path) -> Path:
    """Return the label path, searching by stem in label_dir."""
    direct = label_dir / f"{img_path.stem}.txt"
    if direct.exists():
        return direct
    # label_dir might be a round dataset where images were renamed img_0000001 —
    # in that case the caller should pass the renamed label dir directly and
    # this fallback won't be needed, but we keep it for robustness.
    return direct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True,  help="Path to .pt weights file")
    ap.add_argument("--labels",  required=True,  help="Directory containing GT .txt label files")
    ap.add_argument("--images",  required=True,  help="Directory containing source images")
    ap.add_argument("--out",     default="vis_output", help="Output directory")
    ap.add_argument("--n",       type=int, default=None,
                    help="Number of random images to sample (default: all)")
    ap.add_argument("--image",   default=None,
                    help="Path to a specific image (overrides --images / --n)")
    ap.add_argument("--conf",    type=float, default=0.25,
                    help="Confidence threshold for predictions (default: 0.25)")
    ap.add_argument("--device",  default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.to(args.device)
    class_names: List[str] = list(model.names.values())

    label_dir = Path(args.labels)

    if args.image:
        image_paths = [Path(args.image)]
    else:
        img_dir = Path(args.images)
        exts    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in exts)
        if args.n is not None:
            random.shuffle(image_paths)
            image_paths = image_paths[:args.n]

    print(f"Rendering {len(image_paths)} images → {out_dir}")

    for img_path in image_paths:
        try:
            canvas = _make_side_by_side(img_path, label_dir, model, class_names, args.conf)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_vis.jpg"), canvas,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        except Exception as e:
            print(f"  skipped {img_path.name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()