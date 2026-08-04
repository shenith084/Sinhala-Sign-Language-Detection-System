"""
regenerate_splits.py
====================
Regenerates train/val/test split CSVs from the current raw dataset.
Split ratios: 70% train, 10% val, 20% test (stratified by class).
"""

import csv
import os
import random
from pathlib import Path

# --- Configuration ---
RAW_DIR = Path("data/raw")
SPLITS_DIR = Path("data/splits")
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20
VIDEO_EXTENSIONS = {".mp4", ".mov"}

# 5-class mapping (class_id assigned alphabetically by folder name to ensure consistency)
CLASS_MAP = {
    "Eat":       0,
    "Good":      1,
    "Hello":     2,
    "House":     3,
    "Thank you": 4,
}

random.seed(SEED)

# --- Collect all videos ---
all_samples = []
class_counts = {}

for class_name, class_id in CLASS_MAP.items():
    class_dir = RAW_DIR / class_name
    if not class_dir.exists():
        print(f"WARNING: Folder not found: {class_dir}")
        continue

    videos = [
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]

    class_counts[class_name] = len(videos)
    for video_path in videos:
        rel_path = video_path.as_posix().replace("\\", "/")
        # Normalize to data/raw/... format
        rel_path = str(Path("data/raw") / class_name / video_path.name).replace("\\", "/")
        all_samples.append({
            "video_path": rel_path,
            "class_name": class_name,
            "class_id": class_id
        })

print(f"\n{'='*50}")
print(f"  Total videos found: {len(all_samples)}")
print(f"{'='*50}")
for cls, cnt in sorted(class_counts.items()):
    print(f"  {cls:<12}: {cnt} videos")
print(f"{'='*50}\n")

# --- Stratified split ---
train_samples, val_samples, test_samples = [], [], []

# Group by class
by_class = {}
for s in all_samples:
    cid = s["class_id"]
    by_class.setdefault(cid, []).append(s)

for cid, samples in sorted(by_class.items()):
    random.shuffle(samples)
    n = len(samples)
    n_train = max(1, round(n * TRAIN_RATIO))
    n_val   = max(1, round(n * VAL_RATIO))
    n_test  = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        n_train = n - n_val - n_test

    train_samples.extend(samples[:n_train])
    val_samples.extend(samples[n_train:n_train + n_val])
    test_samples.extend(samples[n_train + n_val:])

# Shuffle each split
random.shuffle(train_samples)
random.shuffle(val_samples)
random.shuffle(test_samples)

print(f"Split sizes:")
print(f"  Train : {len(train_samples)}")
print(f"  Val   : {len(val_samples)}")
print(f"  Test  : {len(test_samples)}")
print()

# --- Write CSVs ---
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
FIELDNAMES = ["video_path", "class_name", "class_id"]

for split_name, split_data in [
    ("train_split", train_samples),
    ("val_split",   val_samples),
    ("test_split",  test_samples),
]:
    out_path = SPLITS_DIR / f"{split_name}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(split_data)
    print(f"  Written: {out_path}  ({len(split_data)} rows)")

print(f"\n[SUCCESS] All splits regenerated successfully!")
print(f"\nPer-class distribution in train split:")
from collections import Counter
train_class_dist = Counter(s["class_name"] for s in train_samples)
for cls, cnt in sorted(train_class_dist.items()):
    print(f"  {cls:<12}: {cnt}")
