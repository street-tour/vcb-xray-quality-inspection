"""
YOLO 4-Class Defect Dataset Augmentation (Albumentations)
--------------------------------------------------------
- Classes: BellowsRingDistortion (0), Bending (1), ContactSeparation (2), FillingDown (3)
- Goal: Build up to TARGET_PER_CLASS (=1000) images per class (including originals)

- **Input (your structure)**
    DATASET_ROOT/
      BellowsRingDistortion/
        img1.jpg
        img1.txt  (YOLO: cls cx cy w h)
      Bending/
        ...
      ContactSeparation/
        ...
      FillingDown/
        ...

- **Output (mirrors same structure)** inside OUTPUT_ROOT:
    OUTPUT_ROOT/
      BellowsRingDistortion/
        *.jpg, *.txt
      ...

Notes
- This script preserves existing images by copying them first, then adds augmented images
  until each class reaches TARGET_PER_CLASS. If you want “exactly” 1000 augmented-only
  samples per class, set COPY_ORIGINALS=False and adjust logic accordingly.
- If your objects are highly orientation-sensitive, consider disabling vertical flips/rotations.
- Requires: albumentations, opencv-python, numpy, tqdm
    pip install albumentations opencv-python numpy tqdm
"""

import os
import cv2
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm
import albumentations as A

# =========================
# Configuration
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# === PATHS ===
# Set your dataset root and output here
DATASET_ROOT = r"D:\"   
OUTPUT_ROOT = r"D:\" 

# === CLASSES ===
# Ensure this matches your training YAML order
CLASSES = [
    "BellowsRingDistortion",    # 0
    "FillingDown",              # 1
    "Bending",                  # 2
    "ContactSeparation"         # 3
]
NAME2ID = {n: i for i, n in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
VALID_CLASS_IDS = set(range(NUM_CLASSES))

# === TARGETS ===
TARGET_PER_CLASS = 1000         # total per class including originals (instance-level)
COPY_ORIGINALS = True           # copy originals into OUTPUT first
MAX_AUG_PER_IMAGE = 10          # safety cap: max augmented variants generated from a single source image per loop pass

# === LABEL SANITIZATION ===
# If your legacy labels have different IDs, remap them here (e.g., {4: 2})
CLASS_ID_REMAP: Dict[int, int] = {}  # e.g., {4: 2}
DROP_UNKNOWN_CLASS = True            # drop boxes whose (remapped) class is not in [0..NUM_CLASSES-1]
LOG_SKIPPED = True                   # log skipped annotations to file
SKIP_LOG_PATH = Path(OUTPUT_ROOT) / "skipped_labels.txt"

# Drop boxes that shrink too much after transforms
MIN_VISIBILITY = 0.2            # Albumentations visibility threshold (fraction of box area remaining)

# Valid image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Helpers for output dirs per class
def out_class_dir(class_name: str) -> Path:
    return Path(OUTPUT_ROOT) / class_name

def ensure_out_dirs_for_classes():
    for c in CLASSES:
        (out_class_dir(c)).mkdir(parents=True, exist_ok=True)

# =========================
# Utilities
# =========================

# --- Logging skipped annotations/files
_SKIPPED_LINES = []


def _log_skip(msg: str):
    if LOG_SKIPPED:
        _SKIPPED_LINES.append(msg)


def scan_label_space(root: Path) -> Dict[int, int]:
    """Scan all .txt under class folders and count raw class IDs found (without geometry parsing)."""
    counts: Dict[int, int] = {}
    for class_name in CLASSES:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for lbl_path in class_dir.glob("**/*.txt"):
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            raw_cls = int(parts[0])
                        except Exception:
                            _log_skip(f"PARSE_ERR {lbl_path}#{i}: {line.strip()}")
                            continue
                        counts[raw_cls] = counts.get(raw_cls, 0) + 1
            except Exception as e:
                _log_skip(f"READ_ERR {lbl_path}: {e}")
    return counts


def yolo_to_albu_bbox(line: str, img_w: int, img_h: int) -> Tuple[float, float, float, float, int]:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO label line: {line}")
    cls = int(parts[0])
    cx, cy, w, h = map(float, parts[1:])

    # Optional remap
    if cls in CLASS_ID_REMAP:
        cls = CLASS_ID_REMAP[cls]

    # Convert to Pascal VOC (x_min, y_min, x_max, y_max)
    x_min = (cx - w/2) * img_w
    y_min = (cy - h/2) * img_h
    x_max = (cx + w/2) * img_w
    y_max = (cy + h/2) * img_h
    return x_min, y_min, x_max, y_max, cls


def albu_to_yolo_bbox(x_min, y_min, x_max, y_max, img_w, img_h):
    # Clamp
    x_min = max(0, min(x_min, img_w - 1))
    y_min = max(0, min(y_min, img_h - 1))
    x_max = max(0, min(x_max, img_w - 1))
    y_max = max(0, min(y_max, img_h - 1))
    bw = x_max - x_min
    bh = y_max - y_min
    if bw <= 1e-6 or bh <= 1e-6:
        return None
    cx = x_min + bw/2
    cy = y_min + bh/2
    return cx / img_w, cy / img_h, bw / img_w, bh / img_h


def read_yolo_labels(label_path: Path, img_w: int, img_h: int):
    boxes = []
    classes = []
    if not label_path.exists():
        return boxes, classes
    with open(label_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                x_min, y_min, x_max, y_max, cls = yolo_to_albu_bbox(line, img_w, img_h)
            except Exception:
                _log_skip(f"PARSE_ERR {label_path}#{i}: {line.strip()}")
                continue
            # Filter/Drop unknown classes after remap
            if cls not in VALID_CLASS_IDS:
                if DROP_UNKNOWN_CLASS:
                    _log_skip(f"DROP_UNKNOWN {label_path}#{i}: cls={cls}")
                    continue
            boxes.append([x_min, y_min, x_max, y_max])
            classes.append(cls)
    return boxes, classes


def write_yolo_labels(save_path: Path, bboxes: List[List[float]], classes: List[int], img_w: int, img_h: int):
    lines = []
    for (x_min, y_min, x_max, y_max), cls in zip(bboxes, classes):
        yolo_vals = albu_to_yolo_bbox(x_min, y_min, x_max, y_max, img_w, img_h)
        if yolo_vals is None:
            continue
        cx, cy, w, h = yolo_vals
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    if lines:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


def list_images_by_class(root: Path):
    """Return dict[class_id] -> list of (img_path, lbl_path) under class-named subfolders."""
    result = {i: [] for i in range(NUM_CLASSES)}
    for class_name in CLASSES:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for p in class_dir.glob("**/*"):
            if p.suffix.lower() in IMG_EXTS:
                img_path = p
                lbl_path = p.with_suffix('.txt')
                result[NAME2ID[class_name]].append((img_path, lbl_path))
    return result

# =========================
# Augmentation Pipelines
# =========================
# Multiple recipes sampled randomly for diversity
basic_geom = A.OneOf([
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.Affine(rotate=(-15, 15), translate_percent=(0.0, 0.05), scale=(0.9, 1.1), cval=0, mode=cv2.BORDER_REFLECT_101, p=1.0),
], p=1.0)

photometric = A.OneOf([
    A.RandomBrightnessContrast(p=1.0),
    A.CLAHE(p=1.0),
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
    A.HueSaturationValue(p=1.0)
], p=1.0)

# Fix: GaussianBlur uses blur_limit, GaussNoise uses var_limit
noise_blur = A.OneOf([
    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    A.MotionBlur(blur_limit=(3, 7), p=1.0),
    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)
], p=1.0)

# Some albumentations versions differ on CoarseDropout signature.
# Try to create it; if it fails, fall back to disabling it.
try:
    cutout_like = A.CoarseDropout(
        max_holes=3, max_height=48, max_width=48,
        min_holes=1, min_height=16, min_width=16,
        fill_value=0, p=0.5
    )
    HAS_CUTOUT = True
except Exception:
    HAS_CUTOUT = False

# Compose a set of candidate pipelines
PIPELINES = [
    A.Compose([basic_geom], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"], min_visibility=MIN_VISIBILITY)),
    A.Compose([photometric], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"], min_visibility=MIN_VISIBILITY)),
    A.Compose([noise_blur], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"], min_visibility=MIN_VISIBILITY)),
    A.Compose([basic_geom, photometric], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"], min_visibility=MIN_VISIBILITY)),
    A.Compose([basic_geom, noise_blur], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"], min_visibility=MIN_VISIBILITY)),
    A.Compose([photometric, noise_blur], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"], min_visibility=MIN_VISIBILITY)),
]
if HAS_CUTOUT:
    PIPELINES.append(
        A.Compose([basic_geom, photometric, noise_blur, cutout_like],
                  bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"], min_visibility=MIN_VISIBILITY))
    )

# =========================
# Core Logic
# =========================

def copy_originals_and_count_per_class(imgs_by_class: dict):
    """Copy originals into OUTPUT and return per-class counts based on labels (instance-level)."""
    per_class_counts = {i: 0 for i in range(NUM_CLASSES)}
    for cls_id, pairs in imgs_by_class.items():
        cls_name = CLASSES[cls_id]
        out_dir = out_class_dir(cls_name)
        for img_path, lbl_path in tqdm(pairs, desc=f"Copy originals [{cls_name}]"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            boxes, classes = read_yolo_labels(lbl_path, w, h)
            # If all boxes were dropped, skip this image
            if not classes:
                _log_skip(f"NO_VALID_BOXES {lbl_path}")
                continue
            if COPY_ORIGINALS:
                dst_img = out_dir / img_path.name
                dst_lbl = out_dir / (img_path.stem + ".txt")
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dst_img)
                write_yolo_labels(dst_lbl, boxes, classes, w, h)
            for c in classes:
                if c in per_class_counts:
                    per_class_counts[c] += 1
                else:
                    _log_skip(f"COUNT_SKIP cls={c} in {lbl_path}")
    return per_class_counts


def choose_pipeline():
    return random.choice(PIPELINES)


def next_filename(base_dir: Path, stem: str, ext: str) -> Path:
    """Generate a unique filename like stem_aug_000123.ext to avoid collisions."""
    i = 0
    while True:
        candidate = base_dir / f"{stem}_aug_{i:06d}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def augment_until_target(imgs_by_class: dict, starting_counts: dict):
    per_class_counts = starting_counts.copy()

    # Build a flat pool with majority-class tagging to help balance
    by_majority_class = {i: [] for i in range(NUM_CLASSES)}
    for cls_id, pairs in imgs_by_class.items():
        for img_path, lbl_path in pairs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            boxes, classes = read_yolo_labels(lbl_path, w, h)
            if not classes:
                continue
            maj = max(set(classes), key=classes.count)
            by_majority_class[maj].append((img_path, lbl_path, boxes, classes, w, h))

    progress = True
    with tqdm(total=NUM_CLASSES * TARGET_PER_CLASS, desc="Augmenting (instance-level)") as pbar:
        initial_sum = sum(starting_counts.values())
        pbar.update(initial_sum)

        while progress:
            progress = False
            for cls_id in range(NUM_CLASSES):
                if per_class_counts[cls_id] >= TARGET_PER_CLASS:
                    continue
                pool = by_majority_class[cls_id]
                if not pool:
                    continue
                candidates = random.sample(pool, k=min(4, len(pool)))
                for img_path, lbl_path, boxes, classes, w, h in candidates:
                    if per_class_counts[cls_id] >= TARGET_PER_CLASS:
                        break
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    aug_times = 0
                    out_dir = out_class_dir(CLASSES[cls_id])
                    while per_class_counts[cls_id] < TARGET_PER_CLASS and aug_times < MAX_AUG_PER_IMAGE:
                        aug = choose_pipeline()
                        transformed = aug(image=img, bboxes=boxes, category_ids=classes)
                        img_aug = transformed["image"]
                        bboxes_aug = transformed["bboxes"]
                        classes_aug = transformed["category_ids"]
                        if not bboxes_aug or (cls_id not in classes_aug):
                            aug_times += 1
                            continue
                        out_img_path = next_filename(out_dir, img_path.stem, img_path.suffix)
                        out_lbl_path = out_img_path.with_suffix('.txt')
                        cv2.imwrite(str(out_img_path), img_aug)
                        H, W = img_aug.shape[:2]
                        write_yolo_labels(out_lbl_path, bboxes_aug, classes_aug, W, H)
                        for c in classes_aug:
                            if c in per_class_counts:
                                per_class_counts[c] += 1
                                pbar.update(1)
                            else:
                                _log_skip(f"AUG_COUNT_SKIP cls={c} from {lbl_path}")
                        aug_times += 1
                        progress = True
            if all(per_class_counts[c] >= TARGET_PER_CLASS for c in range(NUM_CLASSES)):
                break
    return per_class_counts

# =========================
# Main
# =========================

def main():
    print("Ensuring output class dirs...")
    ensure_out_dirs_for_classes()

    print("Scanning label space (raw class IDs)...")
    id_counts = scan_label_space(Path(DATASET_ROOT))
    print("Found class IDs in labels:", id_counts)
    unknown = {k: v for k, v in id_counts.items() if (CLASS_ID_REMAP.get(k, k) not in VALID_CLASS_IDS)}
    if unknown:
        print("WARNING: Unknown/Out-of-range class IDs after remap:", unknown)
        if not DROP_UNKNOWN_CLASS:
            print("Set DROP_UNKNOWN_CLASS=True or configure CLASS_ID_REMAP to handle them.")

    print("Listing source images by class...")
    imgs_by_class = list_images_by_class(Path(DATASET_ROOT))

    total_imgs = sum(len(v) for v in imgs_by_class.values())
    assert total_imgs > 0, f"No images found under class folders in {DATASET_ROOT}"

    print("Copying originals & counting per-class instances (from labels)...")
    starting_counts = copy_originals_and_count_per_class(imgs_by_class) if COPY_ORIGINALS else {i: 0 for i in range(NUM_CLASSES)}
    print("Starting counts:", starting_counts)

    print("Augmenting until per-class targets are met...")
    final_counts = augment_until_target(imgs_by_class, starting_counts)
    print("Final counts:", final_counts)

    # Flush skip log
    if LOG_SKIPPED and _SKIPPED_LINES:
        SKIP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SKIP_LOG_PATH, "w", encoding="utf-8") as f:
            for line in _SKIPPED_LINES:
                f.write(line + "\n")
        print("Wrote skip log to:", SKIP_LOG_PATH)

    summary = {
        "classes": {i: name for i, name in enumerate(CLASSES)},
        "starting_counts": starting_counts,
        "final_counts": final_counts,
        "target_per_class": TARGET_PER_CLASS,
        "copy_originals": COPY_ORIGINALS,
        "seed": SEED,
        "input_structure": "class-subfolder",
        "raw_label_id_counts": id_counts,
        "unknown_after_remap": unknown,
        "drop_unknown": DROP_UNKNOWN_CLASS,
        "class_id_remap": CLASS_ID_REMAP,
        "skip_log_path": str(SKIP_LOG_PATH) if LOG_SKIPPED else None,
    }
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    with open(Path(OUTPUT_ROOT) / "augmentation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Saved summary to", Path(OUTPUT_ROOT) / "augmentation_summary.json")


if __name__ == "__main__":
    main()
