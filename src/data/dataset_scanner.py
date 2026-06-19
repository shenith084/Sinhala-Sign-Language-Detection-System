"""
dataset_scanner.py
==================
Phase 0, Step 0.5 — Dataset Structure Scanner

Scans the SSL400 'Dataset - Original' folder (which uses English category/word
folder names) and:
  1. Enumerates every leaf folder as a class, assigns numeric IDs 0..N-1
  2. Counts videos per class (supports .mov, .mp4, .avi)
  3. Writes data/splits/sinhala_word_map.csv
  4. Writes data/splits/dataset_summary.csv
  5. Prints a full distribution report

Usage:
    python dataset_scanner.py
"""

import os
import csv
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import yaml

# ── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Load Config ──────────────────────────────────────────────────────────────
CONFIG_PATH = Path("config.yaml")


def load_config() -> dict:
    """Load central YAML configuration file."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Supported Video Extensions ───────────────────────────────────────────────
VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}


def scan_dataset(raw_source: str) -> List[Dict]:
    """
    Walk the SSL400 dataset directory and collect all class entries.

    The dataset is structured as:
        raw_source/
          <Category>/
            <WordClass>/
              video_001.mov
              ...

    Args:
        raw_source: Path to 'SSL400/Dataset - Original'

    Returns:
        List of dicts with keys:
          class_id, category, class_name_english, video_count, video_paths
    """
    source_path = Path(raw_source)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Dataset source not found: {source_path.resolve()}\n"
            "Please ensure the SSL400 folder is inside the project directory."
        )

    class_entries: List[Dict] = []
    class_id = 0

    # Walk: Category level → Word level (leaf)
    categories = sorted([d for d in source_path.iterdir() if d.is_dir()])

    for category_dir in categories:
        word_dirs = sorted([d for d in category_dir.iterdir() if d.is_dir()])

        for word_dir in word_dirs:
            videos = [
                f for f in word_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
            ]

            class_entries.append({
                "class_id": class_id,
                "category": category_dir.name,
                "class_name_english": word_dir.name,
                "class_name_sinhala": f"Class_{class_id}",   # TODO: fill Sinhala translations
                "video_count": len(videos),
                "video_paths": [str(v) for v in sorted(videos)],
                "folder_path": str(word_dir),
            })
            class_id += 1

    return class_entries


def print_distribution_report(class_entries: List[Dict]) -> None:
    """Print a detailed class distribution report to the console."""
    counts = [e["video_count"] for e in class_entries]
    total_videos = sum(counts)
    num_classes = len(class_entries)

    logger.info("=" * 60)
    logger.info("  SSL400 DATASET SCAN REPORT")
    logger.info("=" * 60)
    logger.info(f"  Total classes found   : {num_classes}")
    logger.info(f"  Total videos found    : {total_videos}")
    logger.info(f"  Min videos per class  : {min(counts)}")
    logger.info(f"  Max videos per class  : {max(counts)}")
    logger.info(f"  Mean videos per class : {sum(counts)/num_classes:.2f}")
    logger.info(f"  Std dev               : {compute_std(counts):.2f}")
    logger.info("=" * 60)

    # Categories breakdown
    from collections import defaultdict
    cat_counts: Dict[str, int] = defaultdict(int)
    for e in class_entries:
        cat_counts[e["category"]] += 1

    logger.info("\n  Classes per category:")
    for cat, cnt in sorted(cat_counts.items()):
        logger.info(f"    {cat:<25} {cnt:>4} classes")

    # Flag sparse classes
    sparse = [e for e in class_entries if e["video_count"] < 8]
    if sparse:
        logger.warning(f"\n  ⚠  {len(sparse)} classes have < 8 videos (high overfitting risk):")
        for e in sparse:
            logger.warning(
                f"    [{e['class_id']:>3}] {e['category']}/{e['class_name_english']} "
                f"— {e['video_count']} videos"
            )
    else:
        logger.info("\n  ✅ All classes have >= 8 videos.")


def compute_std(values: List[int]) -> float:
    """Compute standard deviation of a list of integers."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return variance ** 0.5


def write_word_map(class_entries: List[Dict], output_path: Path) -> None:
    """
    Write sinhala_word_map.csv mapping class_id → names.

    Columns: class_id, category, class_name_english, class_name_sinhala

    Args:
        class_entries: Output from scan_dataset()
        output_path:   Destination CSV path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "category", "class_name_english",
                        "class_name_sinhala"],
        )
        writer.writeheader()
        for e in class_entries:
            writer.writerow({
                "class_id": e["class_id"],
                "category": e["category"],
                "class_name_english": e["class_name_english"],
                "class_name_sinhala": e["class_name_sinhala"],
            })
    logger.info(f"  ✅ Saved word map  → {output_path}")


def write_dataset_summary(class_entries: List[Dict], output_path: Path) -> None:
    """
    Write dataset_summary.csv with per-class video counts.

    Columns: class_id, category, class_name_english, video_count, folder_path

    Args:
        class_entries: Output from scan_dataset()
        output_path:   Destination CSV path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "category", "class_name_english",
                        "video_count", "folder_path"],
        )
        writer.writeheader()
        for e in class_entries:
            writer.writerow({
                "class_id": e["class_id"],
                "category": e["category"],
                "class_name_english": e["class_name_english"],
                "video_count": e["video_count"],
                "folder_path": e["folder_path"],
            })
    logger.info(f"  ✅ Saved summary   → {output_path}")


def write_video_index(class_entries: List[Dict], output_path: Path) -> None:
    """
    Write a full video index CSV listing every individual video with its label.

    Columns: class_id, category, class_name_english, video_path

    This is used by generate_splits.py to create stratified splits.

    Args:
        class_entries: Output from scan_dataset()
        output_path:   Destination CSV path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "category", "class_name_english", "video_path"],
        )
        writer.writeheader()
        for e in class_entries:
            for vp in e["video_paths"]:
                writer.writerow({
                    "class_id": e["class_id"],
                    "category": e["category"],
                    "class_name_english": e["class_name_english"],
                    "video_path": vp,
                })
    logger.info(f"  ✅ Saved video index → {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    """Run the full dataset scan and write all output files."""
    config = load_config()

    raw_source = config["dataset"]["raw_source"]
    word_map_path = Path(config["dataset"]["word_map"])
    summary_path = Path(config["dataset"]["summary"])
    video_index_path = Path("data/splits/video_index.csv")

    logger.info(f"Scanning dataset at: {Path(raw_source).resolve()}")

    try:
        class_entries = scan_dataset(raw_source)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Write outputs
    print_distribution_report(class_entries)
    write_word_map(class_entries, word_map_path)
    write_dataset_summary(class_entries, summary_path)
    write_video_index(class_entries, video_index_path)

    # Print the num_classes for config reference
    num_classes = len(class_entries)
    logger.info("")
    logger.info(f"  📌 NUM CLASSES = {num_classes}")
    logger.info(f"  Update config.yaml → model.num_classes: {num_classes}")
    logger.info("  Phase 0 dataset scan complete ✅")


if __name__ == "__main__":
    main()
