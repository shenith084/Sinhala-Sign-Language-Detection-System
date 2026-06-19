"""
train.py
========
Master Training Script

Trains the I3D model for a specified experiment ID using a single-phase
feature extraction strategy:
  Phase 1 (Feature Extraction): Backbone FROZEN, LR=1e-3, custom head trained.

The ONLY variable between experiments is the image enhancement function.
All other hyperparameters, splits, and seeds are identical.

Usage (laptop CPU — small scale test):
    python src/training/train.py --exp_id 1

Usage (Colab / GPU — full training):
    python src/training/train.py --exp_id 1 --batch_size 8 --full

Arguments:
    --exp_id      Experiment ID 1–5 (required)
    --batch_size  Batch size (default: 4 for CPU, 8 for GPU)
    --full        If set, runs full epoch counts. Otherwise runs quick test.
    --config      Path to config.yaml (default: config.yaml)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

# ── Environment Setup ─────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "2"          # Suppress TF C++ logs
os.environ["TF_ENABLE_ONEDNN_OPTS"]   = "0"          # Suppress oneDNN warnings
os.environ["PYTHONIOENCODING"]         = "utf-8"      # Fix Windows console encoding

# ── Path Setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.i3d_builder import (
    build_and_compile_phase1,
    compile_model,
)
from src.data.tf_dataset_builder import build_dataset
from src.training.callbacks import build_callbacks

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seeds(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_phase1(
    exp_id: int,
    config: dict,
    batch_size: int,
    max_epochs: int,
    config_path: str,
) -> tf.keras.Model:
    """
    Phase 1 Warm-Up: Train classification head only (backbone frozen).

    Args:
        exp_id:       Experiment ID (1–5)
        config:       Loaded config dictionary
        batch_size:   Batch size for DataLoader
        max_epochs:   Maximum training epochs for Phase 1
        config_path:  Path to config.yaml

    Returns:
        Trained model after Phase 1
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  EXPERIMENT {exp_id} — PHASE 1 (WARM-UP)")
    logger.info(f"  Enhancement: {config['experiments'][exp_id]['name']}")
    logger.info("=" * 60)

    num_classes   = config["model"]["num_classes"]
    target_frames = config["video"]["target_frames"]
    seed          = config["project"]["seed"]

    # ── Build Datasets ────────────────────────────────────────────────────────
    logger.info("Building training dataset...")
    train_ds = build_dataset(
        split_csv="data/splits/train_split.csv",
        exp_id=exp_id,
        batch_size=batch_size,
        num_classes=num_classes,
        target_frames=target_frames,
        augment=True,
        shuffle=True,
        seed=seed,
        config_path=config_path,
    )

    logger.info("Building validation dataset...")
    val_ds = build_dataset(
        split_csv="data/splits/val_split.csv",
        exp_id=exp_id,
        batch_size=batch_size,
        num_classes=num_classes,
        target_frames=target_frames,
        augment=False,
        shuffle=False,
        seed=seed,
        config_path=config_path,
    )

    # ── Build Model ───────────────────────────────────────────────────────────
    logger.info("Building I3D model (backbone frozen)...")
    model = build_and_compile_phase1(config_path=config_path)

    # ── Build Callbacks ───────────────────────────────────────────────────────
    callbacks = build_callbacks(exp_id=exp_id, phase=1, config_path=config_path)

    # ── Train Phase 1 ─────────────────────────────────────────────────────────
    logger.info(f"Starting Phase 1 training (max {max_epochs} epochs)...")
    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1,  # Show progress bar
    )

    duration_min = (time.time() - start_time) / 60
    best_val_acc = max(history.history.get("val_accuracy", [0]))

    logger.info("")
    logger.info(f"  Training Complete:")
    logger.info(f"  Duration      : {duration_min:.1f} minutes")
    logger.info(f"  Best val_acc  : {best_val_acc:.4f}")
    logger.info(f"  Epochs run    : {len(history.history['loss'])}")

    # ── Save Final Model ──────────────────────────────────────────────────────
    exp_config = config["experiments"][exp_id]
    final_path = Path(exp_config["model_dir"]) / "best_model.keras"
    model.save(str(final_path))
    logger.info(f"  Final model saved → {final_path}")

    return model



def main() -> None:
    """Main entry point for training pipeline."""
    parser = argparse.ArgumentParser(
        description="SSL400 I3D Training Script"
    )
    parser.add_argument(
        "--exp_id", type=int, required=True,
        choices=[1, 2, 3, 4, 5],
        help="Experiment ID (1=Baseline, 2=CLAHE, 3=Bilateral, 4=Unsharp, 5=Hybrid)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size (4 for CPU/laptop, 8 for GPU)"
    )

    parser.add_argument(
        "--full", action="store_true",
        help="Run full epoch counts (default: quick test with 3 epochs per phase)"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["project"]["seed"]
    set_global_seeds(seed)

    if args.full:
        max_epochs = 50 # Increased max epochs since it's the only phase
    else:
        max_epochs = 3   # Quick sanity check for laptop

    logger.info("")
    logger.info("SSL400 Sinhala Sign Language Research — Training")
    logger.info(f"Experiment     : EXP{args.exp_id} — {config['experiments'][args.exp_id]['name']}")
    logger.info(f"Batch size     : {args.batch_size}")
    logger.info(f"Mode           : {'FULL TRAINING' if args.full else 'QUICK TEST (3 epochs)'}")
    logger.info(f"Seed           : {seed}")
    logger.info(f"TF version     : {tf.__version__}")
    logger.info(f"GPU available  : {len(tf.config.list_physical_devices('GPU')) > 0}")
    logger.info("")

    model = None

    # ── Train ───────────────────────────────────────────────────────────────
    model = run_phase1(
        exp_id=args.exp_id,
        config=config,
        batch_size=args.batch_size,
        max_epochs=max_epochs,
        config_path=args.config,
    )

    logger.info("")
    logger.info(f"Training for EXP{args.exp_id} complete.")
    logger.info("Next: run evaluation with: python src/evaluation/evaluate.py --exp_id N")


if __name__ == "__main__":
    main()
