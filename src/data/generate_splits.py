"""
generate_splits.py
==================
Phase 1, Step 1.1 — Stratified Train/Val/Test Split Generator

Reads the video_index.csv (produced by dataset_scanner.py) and creates
stratified 70/10/20 splits. The resulting CSVs are shared identically
across ALL 5 experiments — only the enhancement function changes per
experiment, not the data split.

Usage:
    python src/data/generate_splits.py
"""

import logging
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_splits(
    video_index_path: Path,
    splits_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> None:
    """
    Create stratified train/val/test splits from a video index CSV.

    Strategy:
        1. First split: (train+val) vs test  at ratio (80% vs 20%)
        2. Second split: train vs val         at ratio (87.5% vs 12.5%)
           → 70% train, 10% val of total

    The same indices are reused identically across all 5 experiments.

    Args:
        video_index_path: Path to data/splits/video_index.csv
        splits_dir:       Output directory for split CSVs
        train_ratio:      Fraction of data for training (default 0.70)
        val_ratio:        Fraction of data for validation (default 0.10)
        test_ratio:       Fraction of data for testing (default 0.20)
        seed:             Random seed for reproducibility (default 42)

    Raises:
        FileNotFoundError: If video_index.csv does not exist
        ValueError:        If ratios do not sum to 1.0
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Train/val/test ratios must sum to 1.0"

    if not video_index_path.exists():
        raise FileNotFoundError(
            f"Video index not found: {video_index_path}\n"
            "Run dataset_scanner.py first."
        )

    logger.info(f"Loading video index from: {video_index_path}")
    df = pd.read_csv(video_index_path)
    total = len(df)
    num_classes = df["class_id"].nunique()

    logger.info(f"  Total videos : {total}")
    logger.info(f"  Num classes  : {num_classes}")

    # ── Filter out classes with only 1 video (can't stratify) ────────────────
    class_counts = df["class_id"].value_counts()
    single_video_classes = class_counts[class_counts < 2].index.tolist()

    if single_video_classes:
        logger.warning(
            f"  ⚠  {len(single_video_classes)} classes have only 1 video — "
            "they will go entirely into training (cannot stratify)."
        )
        single_df = df[df["class_id"].isin(single_video_classes)].copy()
        df = df[~df["class_id"].isin(single_video_classes)].copy()
    else:
        single_df = pd.DataFrame()

    # ── Split 1: (train+val) vs test ─────────────────────────────────────────
    trainval_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["class_id"],
        random_state=seed,
    )

    # ── Split 2: train vs val ─────────────────────────────────────────────────
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_ratio_adjusted,
        stratify=trainval_df["class_id"],
        random_state=seed,
    )

    # ── Merge single-video classes into training ──────────────────────────────
    if not single_df.empty:
        train_df = pd.concat([train_df, single_df], ignore_index=True)

    # ── Save splits ───────────────────────────────────────────────────────────
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train_split.csv"
    val_path   = splits_dir / "val_split.csv"
    test_path  = splits_dir / "test_split.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test_df.to_csv(test_path,   index=False)

    logger.info("")
    logger.info("  Split Results:")
    logger.info(f"  ├── Train : {len(train_df):>5} videos  ({len(train_df)/total*100:.1f}%)")
    logger.info(f"  ├── Val   : {len(val_df):>5} videos  ({len(val_df)/total*100:.1f}%)")
    logger.info(f"  └── Test  : {len(test_df):>5} videos  ({len(test_df)/total*100:.1f}%)")
    logger.info("")
    logger.info(f"  ✅ Saved → {train_path}")
    logger.info(f"  ✅ Saved → {val_path}")
    logger.info(f"  ✅ Saved → {test_path}")

    # ── Verify class coverage ─────────────────────────────────────────────────
    train_classes = set(train_df["class_id"].unique())
    val_classes   = set(val_df["class_id"].unique())
    test_classes  = set(test_df["class_id"].unique())
    all_classes   = set(df["class_id"].unique()) | set(single_df["class_id"].unique() if not single_df.empty else [])

    logger.info(f"  Class coverage — Train: {len(train_classes)}, "
                f"Val: {len(val_classes)}, Test: {len(test_classes)}, "
                f"Total: {len(all_classes)}")

    missing_from_val = all_classes - val_classes
    if missing_from_val:
        logger.warning(
            f"  ⚠  {len(missing_from_val)} classes missing from val set "
            "(expected for very sparse classes)."
        )

    logger.info("  Phase 1 splits generation complete ✅")


def main() -> None:
    """Entry point for split generation."""
    config = load_config()

    video_index_path = Path("data/splits/video_index.csv")
    splits_dir       = Path(config["dataset"]["splits_dir"])
    seed             = config["project"]["seed"]
    train_ratio      = config["splits"]["train"]
    val_ratio        = config["splits"]["val"]
    test_ratio       = config["splits"]["test"]

    generate_splits(
        video_index_path=video_index_path,
        splits_dir=splits_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


if __name__ == "__main__":
    main()
