from __future__ import annotations

import csv
import gc
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ssl_yolo")


@dataclass
class AugParams:
    """Ultralytics augmentation kwargs passed directly to YOLO.train()."""

    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4

    degrees: float     = 0.0
    translate: float   = 0.1
    scale: float       = 0.5
    shear: float       = 0.0
    perspective: float = 0.0
    flipud: float      = 0.0
    fliplr: float      = 0.5

    mosaic: float     = 1.0
    mixup: float      = 0.0
    copy_paste: float = 0.0
    erasing: float    = 0.0

    def to_kwargs(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def teacher_aug_params() -> AugParams:
    """Minimal augmentation for the teacher — mild colour jitter only.

    All geometric transforms are off so that box coordinates stay accurate.
    Localisation error in the merged boxes is the dominant noise source and
    shouldn't be amplified here.
    """
    return AugParams(
        hsv_h=0.010, hsv_s=0.4, hsv_v=0.3,
        degrees=0.0, translate=0.0, scale=0.3,
        shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0,
        mosaic=0.0,
        mixup=0.0, copy_paste=0.0,
        erasing=0.0,
    )


def student_aug_params() -> AugParams:
    """Strong augmentation for the student (Noisy Student / STAC style).

    The student trains against merged labels under heavy distortion, which
    prevents memorisation and forces generalisation.  erasing=0.4 mirrors
    the Cutout variant highlighted in the STAC paper.
    """
    return AugParams(
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        perspective=0.0005,
        flipud=0.05,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        erasing=0.4,
    )


@dataclass
class Config:
    # paths
    data_yaml: str  = "dataset/data.yaml"
    output_dir: str = "ssl_output"

    # model
    model_size: str         = "n"
    pretrained_weights: str = ""

    # training
    epochs_warmup: int  = 50
    epochs_student: int = 30
    patience_warmup: int = 15
    patience_student: int = 15
    batch: int          = 16
    imgsz: int          = 640
    device: str         = "0"

    # noise asymmetry — set False for ablation
    use_strong_aug: bool = True

    # iterative refinement schedule
    rounds: int = 4

    # confidence threshold for merging extra boxes (linearly annealed)
    conf_threshold_start: float = 0.80
    conf_threshold_end: float   = 0.55

    # box sanity filters
    min_box_area_frac: float = 0.002
    max_box_area_frac: float = 0.95
    min_aspect_ratio: float  = 0.1
    max_aspect_ratio: float  = 10.0

    # EMA teacher update (set ema_decay=0.0 for plain checkpoint swap)
    use_ema: bool    = True
    ema_decay: float = 0.9996

    # annotation gap detection
    suspect_conf: float        = 0.75
    suspect_iou_overlap: float = 0.3


class AnnotationRefiner:
    """Finds boxes the teacher predicts with high confidence that have no
    matching GT annotation, and returns them as extra label lines to merge
    into the existing label files.

    Every image stays in the training set with its original GT boxes intact —
    the extras are appended on top.  Inference always runs with augment=False
    so that the merged coordinates are as accurate as possible.
    """

    def __init__(self, cfg: Config, model_path: str, label_map: Dict[Path, Path]):
        self.cfg       = cfg
        self.model     = YOLO(model_path)
        self.label_map = label_map

    def _load_gt_boxes(self, label_path: Path) -> np.ndarray:
        if not label_path.exists():
            return np.zeros((0, 4), dtype=np.float32)
        rows = []
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                rows.append([float(x) for x in parts[1:5]])
        return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 4))

    @staticmethod
    def _max_iou(box: np.ndarray, gt_boxes: np.ndarray) -> float:
        if len(gt_boxes) == 0:
            return 0.0

        def to_xyxy(b):
            return np.stack([
                b[..., 0] - b[..., 2] / 2,
                b[..., 1] - b[..., 3] / 2,
                b[..., 0] + b[..., 2] / 2,
                b[..., 1] + b[..., 3] / 2,
            ], axis=-1)

        b  = to_xyxy(box)
        gt = to_xyxy(gt_boxes)
        ix1 = np.maximum(b[0], gt[:, 0])
        iy1 = np.maximum(b[1], gt[:, 1])
        ix2 = np.minimum(b[2], gt[:, 2])
        iy2 = np.minimum(b[3], gt[:, 3])
        inter   = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
        area_b  = (b[2] - b[0]) * (b[3] - b[1])
        area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
        return float(np.max(inter / (area_b + area_gt - inter + 1e-9)))

    def _box_is_sane(self, w: float, h: float) -> bool:
        area   = w * h
        aspect = w / (h + 1e-9)
        return (
            self.cfg.min_box_area_frac <= area   <= self.cfg.max_box_area_frac and
            self.cfg.min_aspect_ratio  <= aspect <= self.cfg.max_aspect_ratio
        )

    def scan(self, image_paths: List[Path], conf_threshold: float) -> Dict[Path, List[str]]:
        """Return extra YOLO label lines per image for boxes not in the GT.

        Images with no gaps are absent from the returned dict.
        """
        augmented: Dict[Path, List[str]] = {}
        log.info("AnnotationRefiner: scanning %d images (conf>=%.2f) ...",
                 len(image_paths), conf_threshold)

        for i in range(0, len(image_paths), 32):
            batch   = image_paths[i:i + 32]
            results = self.model.predict(
                [str(p) for p in batch],
                conf=conf_threshold,
                verbose=False,
                device=self.cfg.device,
                augment=False,
                stream=True,
            )
            for img_path, result in zip(batch, results):
                label_path = self.label_map.get(img_path)
                if label_path is None or result.boxes is None or len(result.boxes) == 0:
                    continue

                gt_boxes = self._load_gt_boxes(label_path)
                confs    = result.boxes.conf.cpu().numpy()
                cls_ids  = result.boxes.cls.cpu().numpy().astype(int)
                xywhn    = result.boxes.xywhn.cpu().numpy()

                extra = []
                for conf, cls_id, box in zip(confs, cls_ids, xywhn):
                    cx, cy, w, h = box
                    if (self._max_iou(box, gt_boxes) < self.cfg.suspect_iou_overlap
                            and self._box_is_sane(w, h)):
                        extra.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                if extra:
                    augmented[img_path] = extra

        log.info(
            "AnnotationRefiner: found gaps in %d / %d images.",
            len(augmented), len(image_paths),
        )
        return augmented


class IterativeAnnotationTrainer:
    """Iterative annotation refinement loop.

    Each round the current teacher scans all training images for likely
    missing boxes, merges them into the label files, trains a student on
    the result, then updates the teacher via EMA.  No unlabeled pool is
    needed — every image is always trained with its (growing) label set.

    Steps:
      1. Warm-up teacher on original labels with minimal augmentation.
      2. For each round:
         a. Scan for annotation gaps; merge extra boxes into label files.
         b. Train student on all images (with merged labels) using strong aug.
         c. EMA-update teacher from student weights.
      3. Write per-round stats to CSV.
    """

    def __init__(self, cfg: Config):
        self.cfg            = cfg
        self.data_yaml_path = Path(cfg.data_yaml).resolve()
        self.out            = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self.warmup_dir   = self.out / "warmup"
        self.rounds_dir   = self.out / "rounds"
        self.datasets_dir = self.out / "datasets"
        self.stats_csv    = self.out / "round_stats.csv"

        for d in [self.warmup_dir, self.rounds_dir, self.datasets_dir]:
            d.mkdir(parents=True, exist_ok=True)

        with open(self.data_yaml_path) as f:
            self.data_info = yaml.safe_load(f)

        self.nc          = int(self.data_info.get("nc", 1))
        self.class_names = list(self.data_info.get("names", []))

        if self.class_names and len(self.class_names) < self.nc:
            missing = self.nc - len(self.class_names)
            self.class_names.extend(
                [f"class_{i}" for i in range(len(self.class_names), self.nc)]
            )
            log.warning("nc=%d but names has %d entries - padded %d.",
                        self.nc, len(self.class_names) - missing, missing)
        elif self.class_names and len(self.class_names) > self.nc:
            log.warning("nc=%d but names has %d entries - using names length.",
                        self.nc, len(self.class_names))
            self.nc = len(self.class_names)

        if cfg.use_strong_aug:
            self._teacher_aug: Optional[AugParams] = teacher_aug_params()
            self._student_aug: Optional[AugParams] = student_aug_params()
            log.info("Noise asymmetry ON  - teacher=minimal | student=strong")
        else:
            self._teacher_aug = None
            self._student_aug = None
            log.info("Noise asymmetry OFF - Ultralytics defaults (ablation)")

        with open(self.stats_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=[
                "round", "conf_threshold", "images_scanned",
                "images_with_extra_boxes", "extra_boxes_added",
                "map50", "map50_95",
            ]).writeheader()

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _resolve_dataset_entry_path(self, entry: str) -> Path:
        entry_path = Path(entry)
        if entry_path.is_absolute():
            return entry_path

        candidates: List[Path] = []
        data_root = self.data_info.get("path")
        if data_root:
            root = Path(str(data_root))
            if not root.is_absolute():
                root = (self.data_yaml_path.parent / root).resolve()
            candidates.append((root / entry_path).resolve())

        candidates.append((self.data_yaml_path.parent / entry_path).resolve())

        trimmed = entry
        while trimmed.startswith("../"):
            trimmed = trimmed[3:]
        if trimmed:
            candidates.append((self.data_yaml_path.parent / trimmed).resolve())
            if len(self.data_yaml_path.parents) > 1:
                candidates.append((self.data_yaml_path.parents[1] / trimmed).resolve())

        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    def _resolve_split_dir(self, split: str) -> Path:
        val = self.data_info.get(split)
        if val is None:
            if split == "val" and self.data_info.get("valid") is not None:
                val = self.data_info["valid"]
            else:
                raise ValueError(f"Missing split '{split}' in {self.data_yaml_path}")
        if isinstance(val, list):
            raise ValueError(f"Split '{split}' lists are not supported.")
        return self._resolve_dataset_entry_path(str(val))

    @staticmethod
    def _infer_label_path(image_path: Path) -> Path:
        img_str  = str(image_path)
        replaced = img_str.replace("/images/", "/labels/")
        if replaced == img_str and image_path.parent.name == "images":
            return (image_path.parent.parent / "labels" / image_path.name).with_suffix(".txt")
        return Path(replaced).with_suffix(".txt")

    def _collect_images(self, split: str) -> List[Path]:
        img_dir = self._resolve_split_dir(split)
        exts    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in exts)

    def _write_dataset_yaml(
        self,
        name: str,
        images: List[Path],
        augmented_boxes: Optional[Dict[Path, List[str]]] = None,
    ) -> Path:
        """Build a dataset directory with images/ and labels/ in sync.

        If augmented_boxes is provided, extra label lines are appended to each
        image's existing GT file so no original annotations are lost.
        """
        ds_dir = self.datasets_dir / name
        if ds_dir.exists():
            shutil.rmtree(ds_dir)

        imgs_dir   = ds_dir / "train" / "images"
        labels_dir = ds_dir / "train" / "labels"
        imgs_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for idx, img_src in enumerate(images):
            stem      = f"img_{idx:07d}"
            img_dst   = imgs_dir   / f"{stem}{img_src.suffix.lower()}"
            label_dst = labels_dir / f"{stem}.txt"
            label_src = self._infer_label_path(img_src)

            try:
                img_dst.symlink_to(img_src.resolve())
            except OSError:
                shutil.copy2(img_src, img_dst)

            extra = (augmented_boxes or {}).get(img_src, [])
            if label_src.exists():
                existing = label_src.read_text().rstrip("\n")
                label_dst.write_text(
                    (existing + "\n" + "\n".join(extra)).strip() if extra else existing
                )
            else:
                label_dst.write_text("\n".join(extra))

        val_path = self._resolve_split_dir("val")
        ds_yaml  = ds_dir / "dataset.yaml"
        with open(ds_yaml, "w") as f:
            yaml.dump({
                "train": str(imgs_dir.resolve()),
                "val":   str(val_path.resolve()),
                "nc":    self.nc,
                "names": self.class_names,
            }, f)
        return ds_yaml

    # ------------------------------------------------------------------
    # EMA teacher update
    # ------------------------------------------------------------------

    @staticmethod
    def _ema_update(teacher: YOLO, student: YOLO, decay: float) -> None:
        """theta_teacher = decay * theta_teacher + (1 - decay) * theta_student"""
        import torch
        t_state = teacher.model.state_dict()
        s_state = student.model.state_dict()
        for key in t_state:
            if t_state[key].dtype.is_floating_point:
                t_state[key].mul_(decay).add_(s_state[key] * (1.0 - decay))
        teacher.model.load_state_dict(t_state)
        log.info("EMA teacher updated (decay=%.4f)", decay)

    # ------------------------------------------------------------------
    # Training helper
    # ------------------------------------------------------------------

    def _train(
        self,
        model: YOLO,
        data_yaml: Path,
        epochs: int,
        patience: int,
        project: Path,
        name: str,
        aug_params: Optional[AugParams] = None,
    ) -> YOLO:
        aug_kwargs = aug_params.to_kwargs() if aug_params is not None else {}
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            patience=patience,
            batch=self.cfg.batch,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            project=str(project),
            name=name,
            exist_ok=True,
            verbose=False,
            **aug_kwargs,
        )
        save_dir = Path(getattr(model.trainer, "save_dir", project / name))
        best     = save_dir / "weights" / "best.pt"
        if not best.exists():
            fallback = project / name / "weights" / "best.pt"
            if fallback.exists():
                best = fallback
        return YOLO(str(best))

    @staticmethod
    def _free_cuda_memory() -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        cfg          = self.cfg
        train_images = self._collect_images("train")
        log.info("Total train images: %d", len(train_images))

        # 1. Warm-up teacher on original (incomplete) labels
        log.info("=== Warm-up: training initial teacher ===")
        weights     = cfg.pretrained_weights or f"yolo11{cfg.model_size}.pt"
        teacher     = YOLO(weights)
        warmup_yaml = self._write_dataset_yaml("warmup", train_images)
        teacher     = self._train(
            teacher, warmup_yaml,
            epochs=cfg.epochs_warmup,
            patience=cfg.patience_warmup,
            project=self.warmup_dir,
            name="teacher_warmup",
            aug_params=self._teacher_aug,
        )
        warmup_teacher_path = (self.warmup_dir / "teacher_warmup_seed.pt").resolve()
        teacher.save(str(warmup_teacher_path))
        log.info("Warm-up complete -> %s", warmup_teacher_path)

        # 2. Iterative refinement rounds
        label_map  = {p: self._infer_label_path(p) for p in train_images}
        teacher_path = warmup_teacher_path
        all_stats: List[dict] = []

        for r in range(cfg.rounds):
            log.info("=" * 60)
            log.info("ROUND %d / %d", r + 1, cfg.rounds)
            log.info("=" * 60)

            t        = r / max(cfg.rounds - 1, 1)
            conf_thr = cfg.conf_threshold_start + t * (cfg.conf_threshold_end - cfg.conf_threshold_start)

            # 2a. Find annotation gaps
            refiner         = AnnotationRefiner(cfg, str(teacher_path), label_map)
            augmented_boxes = refiner.scan(train_images, conf_thr)
            del refiner
            self._free_cuda_memory()

            extra_box_count = sum(len(v) for v in augmented_boxes.values())
            log.info("Round %d | %d extra boxes across %d images",
                     r, extra_box_count, len(augmented_boxes))
            (self.out / f"augmented_round_{r}.json").write_text(
                json.dumps({str(k): v for k, v in augmented_boxes.items()}, indent=2)
            )

            # 2b. Train student on merged labels
            ds_yaml = self._write_dataset_yaml(f"round_{r}", train_images, augmented_boxes)
            teacher = YOLO(str(teacher_path))
            student = YOLO(str(teacher_path))
            log.info("Round %d | training student", r)
            student = self._train(
                student, ds_yaml,
                epochs=cfg.epochs_student,
                patience=cfg.patience_student,
                project=self.rounds_dir,
                name=f"student_round_{r}",
                aug_params=self._student_aug,
            )

            # 2c. Evaluate on the original val set
            metrics  = student.val(data=str(self.data_yaml_path), verbose=False, device=cfg.device)
            map50    = float(metrics.box.map50)
            map50_95 = float(metrics.box.map)
            log.info("Round %d | mAP50=%.4f  mAP50-95=%.4f", r, map50, map50_95)

            # 2d. Update teacher
            if cfg.use_ema:
                self._ema_update(teacher, student, cfg.ema_decay)
            else:
                teacher = student
                log.info("Teacher updated by student swap (EMA disabled)")

            teacher_path = self.rounds_dir / f"teacher_round_{r}.pt"
            teacher.save(str(teacher_path))
            log.info("Teacher snapshot -> %s", teacher_path)

            stats = {
                "round":                   r,
                "conf_threshold":          conf_thr,
                "images_scanned":          len(train_images),
                "images_with_extra_boxes": len(augmented_boxes),
                "extra_boxes_added":       extra_box_count,
                "map50":                   map50,
                "map50_95":                map50_95,
            }
            all_stats.append(stats)
            with open(self.stats_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=list(stats.keys())).writerow(stats)

            del metrics, student
            self._free_cuda_memory()

        log.info("=" * 60)
        log.info("Done.  Stats -> %s", self.stats_csv)
        for s in all_stats:
            log.info(
                "  Round %d | extra_boxes=%d | mAP50=%.4f | mAP50-95=%.4f",
                s["round"], s["extra_boxes_added"], s["map50"], s["map50_95"],
            )


def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Iterative annotation refinement for partially-annotated YOLO datasets."
    )
    p.add_argument("--data",           default="dataset/data.yaml")
    p.add_argument("--model-size",     default="n", choices=["n", "s", "m", "l", "x"])
    p.add_argument("--output-dir",     default="ssl_output")
    p.add_argument("--rounds",         type=int,   default=4)
    p.add_argument("--epochs-warmup",  type=int,   default=50)
    p.add_argument("--epochs-student", type=int,   default=30)
    p.add_argument("--batch",          type=int,   default=16)
    p.add_argument("--imgsz",          type=int,   default=640)
    p.add_argument("--device",         default="0")
    p.add_argument("--conf-start",     type=float, default=0.80)
    p.add_argument("--conf-end",       type=float, default=0.55)
    p.add_argument("--ema-decay",      type=float, default=0.9996)
    p.add_argument("--no-ema",         action="store_true",
                   help="Disable EMA; use direct student->teacher swap each round.")
    p.add_argument("--pretrained",     default="")
    p.add_argument("--no-strong-aug",  action="store_true",
                   help="Disable aug asymmetry (ablation - both use Ultralytics defaults).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config(
        data_yaml            = args.data,
        model_size           = args.model_size,
        output_dir           = args.output_dir,
        pretrained_weights   = args.pretrained,
        rounds               = args.rounds,
        epochs_warmup        = args.epochs_warmup,
        epochs_student       = args.epochs_student,
        batch                = args.batch,
        imgsz                = args.imgsz,
        device               = args.device,
        conf_threshold_start = args.conf_start,
        conf_threshold_end   = args.conf_end,
        use_ema              = not args.no_ema,
        ema_decay            = args.ema_decay,
        use_strong_aug       = not args.no_strong_aug,
    )
    log.info("Config:\n%s", json.dumps(cfg.__dict__, indent=2))
    IterativeAnnotationTrainer(cfg).run()


if __name__ == "__main__":
    main()