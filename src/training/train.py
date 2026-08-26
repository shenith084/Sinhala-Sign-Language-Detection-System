"""
train.py
========
Master training script for the SSL400 research project.

Runs the full two-phase training protocol for a single experiment:
  Phase 1: Frozen backbone → train LSTM + Dense head (50 epochs max)
  Phase 2: Full fine-tuning → train all layers (30 epochs max)

RESUME SUPPORT:
  If 'models/experiment_{N}/best_model_phase1.keras' already exists,
  Phase 1 is skipped and training resumes directly at Phase 2.
  If 'models/experiment_{N}/best_model_phase2.keras' already exists,
  both phases are skipped (experiment already complete).

  This is critical for free Google Colab where sessions disconnect after ~12 hours.

Usage:
    python src/training/train.py --exp_id 1
    python src/training/train.py --exp_id 2 --batch_size 16  # A100 Colab
    python src/training/train.py --exp_id 3 --drive_dir /content/drive/MyDrive/SSL400_Research
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Critical: Must set env var BEFORE importing tensorflow
# ---------------------------------------------------------------------------
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import tensorflow as tf
tf.config.optimizer.set_jit(False)

try:
    import tf_keras as keras
except ImportError:
    keras = tf.keras

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"All seeds set to: {seed}")


def load_config() -> dict:
    """Load the central config.yaml."""
    with open(PROJECT_ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def get_experiment_config(config: dict, exp_id: int) -> dict:
    """Retrieve experiment-specific config by ID."""
    for exp in config["experiments"]:
        if exp["id"] == exp_id:
            return exp
    raise ValueError(f"Experiment ID {exp_id} not found in config.yaml")


def get_initial_epoch(log_path: str) -> int:
    """Read the CSV log to determine how many epochs were completed."""
    if not os.path.exists(log_path):
        return 0
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            if len(lines) <= 1:
                return 0
            return len(lines) - 1
    except Exception as e:
        logger.warning(f"Failed to read log {log_path} for resume: {e}")
        return 0


def train_phase(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    callbacks: list,
    max_epochs: int,
    phase_name: str,
    initial_epoch: int = 0,
    class_weight: dict = None
) -> keras.callbacks.History:
    """
    Run one training phase and return the history.

    Args:
        model:      Compiled Keras model.
        train_ds:   Training tf.data.Dataset.
        val_ds:     Validation tf.data.Dataset.
        callbacks:  List of Keras callbacks.
        max_epochs: Maximum training epochs for this phase.
        phase_name: Display name ('Phase 1' or 'Phase 2') for logging.
        initial_epoch: Epoch to resume from.

    Returns:
        Keras History object with per-epoch metrics.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  Starting {phase_name}")
    logger.info(f"{'='*60}")
    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=0  # Suppress default verbose; ProgressLogger handles output
    )

    elapsed = (time.time() - start_time) / 60.0
    logger.info(f"  {phase_name} complete in {elapsed:.1f} minutes.")
    logger.info(f"  Best val_accuracy: "
                f"{max(history.history.get('val_accuracy', [0])):.4f}")
    return history


def main(args: argparse.Namespace) -> None:
    """Main training entry point."""
    set_seeds(42)
    config = load_config()
    exp = get_experiment_config(config, args.exp_id)

    logger.info(f"\n{'#'*60}")
    logger.info(f"  EXPERIMENT {args.exp_id}: {exp['name']}")
    logger.info(f"{'#'*60}")

    # Resolve paths
    model_dir = str(PROJECT_ROOT / exp["model_dir"])
    log_dir = str(PROJECT_ROOT / exp["log_dir"])
    # We rely on the Colab notebook to symlink the data directory
    splits_dir = PROJECT_ROOT / config["paths"]["splits"]
    processed_dir = str(PROJECT_ROOT / exp["processed_dir"])

    train_csv = str(splits_dir / "train_split.csv")
    val_csv = str(splits_dir / "val_split.csv")

    num_classes = config["dataset"]["num_classes"]
    num_frames = config["frames"]["num_frames"]
    batch_size = args.batch_size or config["phase1"]["batch_size"]

    # Phase parameters
    p1 = config["phase1"]
    p2 = config["phase2"]

    # Import builders
    if config["model"]["backbone"] == "EfficientNetV2S":
        from models.efficientnet_builder import build_model, compile_phase1, compile_phase2, get_lr_schedule
    elif config["model"]["backbone"] == "MobileNetV3Large":
        from models.backup.mobilenet_builder import build_model, compile_phase1, compile_phase2, get_lr_schedule
    else:
        from models.backup.movinet_builder import build_model, compile_phase1, compile_phase2, load_kinetics_weights, get_lr_schedule
    from data.tf_dataset_builder import build_dataset
    from training.callbacks import get_callbacks, find_resume_checkpoint
    
    # enhance_fn is no longer needed since frames are pre-enhanced in video_to_frames.py

    # Calculate decay steps for learning rate schedules
    import pandas as pd
    try:
        num_train_samples = len(pd.read_csv(train_csv))
    except Exception:
        num_train_samples = 2240 # Fallback 70% of 3200
    steps_per_epoch = max(1, num_train_samples // batch_size)
    decay_steps_p1 = steps_per_epoch * p1["max_epochs"]
    decay_steps_p2 = steps_per_epoch * p2["max_epochs"]

    # BUG FIX: When resuming, the LR schedule must START from the correct
    # step count (not from 0), otherwise the LR is way too high on resume.
    # We compute how many steps were already completed and offset the schedule.
    initial_epoch_p1_pre = get_initial_epoch(str(Path(log_dir) / "training_log_phase1.csv"))
    initial_epoch_p2_pre = get_initial_epoch(str(Path(log_dir) / "training_log_phase2.csv"))
    steps_done_p1 = initial_epoch_p1_pre * steps_per_epoch
    steps_done_p2 = initial_epoch_p2_pre * steps_per_epoch

    lr_schedule_p1 = get_lr_schedule(
        p1["learning_rate"], decay_steps_p1,
        p1.get("lr_schedule", "CosineDecay"),
        initial_step=steps_done_p1
    )
    lr_schedule_p2 = get_lr_schedule(
        p2["learning_rate"], decay_steps_p2,
        p2.get("lr_schedule", "CosineDecay"),
        initial_step=steps_done_p2
    )

    # -------------------------------------------------------------------------
    # COMPUTE CLASS WEIGHTS
    # -------------------------------------------------------------------------
    from sklearn.utils.class_weight import compute_class_weight
    try:
        train_df = pd.read_csv(train_csv)
        y_train = train_df['class_id'].values
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
        logger.info(f"Computed Class Weights: {class_weight_dict}")
    except Exception as e:
        logger.warning(f"Could not compute class weights: {e}")
        class_weight_dict = None

    # -------------------------------------------------------------------------
    # CHECK RESUME STATE
    # -------------------------------------------------------------------------
    phase2_ckpt = find_resume_checkpoint(model_dir, "phase2")
    phase1_ckpt = find_resume_checkpoint(model_dir, "phase1")
    
    # Recover log files from model_dir if they were synced there by GoogleDriveSync
    for p in ["phase1", "phase2"]:
        synced_log = Path(model_dir) / f"training_log_{p}.csv"
        target_log = Path(log_dir) / f"training_log_{p}.csv"
        if synced_log.exists() and not target_log.exists():
            import shutil
            target_log.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(synced_log, target_log)

    log_path_p1 = str(Path(log_dir) / "training_log_phase1.csv")
    log_path_p2 = str(Path(log_dir) / "training_log_phase2.csv")
    
    initial_epoch_p1 = get_initial_epoch(log_path_p1)
    initial_epoch_p2 = get_initial_epoch(log_path_p2)

    if phase2_ckpt:
        if initial_epoch_p2 >= p2["max_epochs"]:
            logger.info(f"✅ Experiment {args.exp_id} Phase 2 checkpoint found. "
                        "Both phases already complete — SKIP.")
            logger.info(f"   Checkpoint: {phase2_ckpt}")
            return
        else:
            logger.info(f"⚠️ Resuming Phase 2 from epoch {initial_epoch_p2}...")

    # -------------------------------------------------------------------------
    # BUILD MODEL
    # -------------------------------------------------------------------------
    logger.info("Building model...")
    model = build_model(
        num_classes=num_classes,
        num_frames=num_frames,
        img_height=config["frames"]["height"],
        img_width=config["frames"]["width"],
        lstm_units=config["model"]["lstm_units"],
        dropout_rate=config["model"]["dropout_rate"]
    )
    model.summary(print_fn=logger.info)

    # -------------------------------------------------------------------------
    # DOWNLOAD KINETICS-600 CHECKPOINT (If MoViNet)
    # -------------------------------------------------------------------------
    if config["model"]["backbone"] not in ["MobileNetV3Large", "EfficientNetV2S"]:
        import tarfile
        ckpt_url = "https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a2_base.tar.gz"
        ckpt_dir = PROJECT_ROOT / "pretrained_weights" / "movinet_a2_base"
        
        if not ckpt_dir.exists():
            logger.info(f"Downloading Kinetics-600 weights from {ckpt_url}...")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            try:
                tar_path = keras.utils.get_file(
                    "movinet_a2_base.tar.gz",
                    ckpt_url,
                    cache_subdir="pretrained_weights",
                    cache_dir=str(PROJECT_ROOT)
                )
                with tarfile.open(tar_path, "r:*") as tar:
                    tar.extractall(path=str(PROJECT_ROOT / "pretrained_weights"))
                logger.info("Weights downloaded and extracted.")
            except Exception as e:
                import shutil
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)
                logger.error(f"Failed to extract weights: {e}")
                raise

    # -------------------------------------------------------------------------
    # PHASE 1 — Warm-up (Frozen Backbone)
    # -------------------------------------------------------------------------
    # CRITICAL FIX: Load Kinetics weights first, then OVERRIDE with any existing
    # checkpoint. Previously, Kinetics weights were loaded after checkpoint weights,
    # which completely wiped out any Phase 1 progress on resume!
    if config["model"]["backbone"] not in ["MobileNetV3Large", "EfficientNetV2S"]:
        if ckpt_dir.exists():
            load_kinetics_weights(model, str(ckpt_dir))

    # BUG FIX: Load Phase 1 checkpoint AFTER Kinetics weights, not before.
    if phase1_ckpt and initial_epoch_p2 == 0:
        logger.info(f"Resuming Phase 1: Loading checkpoint weights from {phase1_ckpt}")
        model.load_weights(phase1_ckpt)
        
    if initial_epoch_p1 < p1["max_epochs"] and initial_epoch_p2 == 0:
        logger.info("\n--- PHASE 1: Frozen backbone warm-up ---")
        model = compile_phase1(
            model,
            learning_rate=lr_schedule_p1,
            num_classes=num_classes,
            label_smoothing=p1["label_smoothing"]
        )

        train_ds = build_dataset(
            split_csv=train_csv,
            processed_dir=processed_dir,
            num_classes=num_classes,
            batch_size=batch_size,
            is_training=True,
            use_augmentation=exp.get("use_augmentation", True),
            use_mixup=(p1["mixup_alpha"] > 0 and exp.get("use_augmentation", True)),
            mixup_alpha=p1["mixup_alpha"],
            num_frames=num_frames,
            target_size=(config["frames"]["width"], config["frames"]["height"]),
            seed=42
        )
        val_ds = build_dataset(
            split_csv=val_csv,
            processed_dir=processed_dir,
            num_classes=num_classes,
            batch_size=batch_size,
            is_training=False,
            num_frames=num_frames,
            target_size=(config["frames"]["width"], config["frames"]["height"])
        )

        callbacks_p1 = get_callbacks(
            exp_id=args.exp_id,
            phase="phase1",
            model_dir=model_dir,
            log_dir=log_dir,
            drive_model_dir=args.drive_dir,
            early_stopping_patience=p1["early_stopping_patience"]
        )

        if phase1_ckpt and initial_epoch_p1 == 0:
            # Only load on fresh start — already handled above for resume
            pass

        train_phase(
            model, train_ds, val_ds, callbacks_p1,
            max_epochs=p1["max_epochs"],
            phase_name="Phase 1 (Frozen Backbone)",
            initial_epoch=initial_epoch_p1,
            class_weight=class_weight_dict
        )

        # Reload best Phase 1 weights after Phase 1 finishes
        phase1_ckpt = find_resume_checkpoint(model_dir, "phase1")

    # Load correct weights before Phase 2:
    # - If resuming Phase 2 mid-run, load the Phase 2 best checkpoint
    # - If starting Phase 2 fresh, load the best Phase 1 checkpoint
    # BUG FIX: Previously the model was compiled BEFORE weights were loaded,
    # which meant the optimizer state was wrong on resume.
    if phase2_ckpt:
        logger.info(f"\n--- Resuming Phase 2: Loading best weights from {phase2_ckpt} ---")
        model.load_weights(phase2_ckpt)
    elif phase1_ckpt:
        logger.info(f"\n--- Starting Phase 2: Loading Phase 1 best weights from {phase1_ckpt} ---")
        model.load_weights(phase1_ckpt)

    # -------------------------------------------------------------------------
    # PHASE 2 — Full Fine-Tuning (All Layers Unfrozen)
    # -------------------------------------------------------------------------
    logger.info("\n--- PHASE 2: Full fine-tuning ---")
    model = compile_phase2(
        model,
        learning_rate=lr_schedule_p2,
        num_classes=num_classes,
        label_smoothing=p2["label_smoothing"]
    )

    # Mixup enabled in Phase 2 for extreme low-resource regularization (if augmentation is enabled)
    train_ds = build_dataset(
        split_csv=train_csv,
        processed_dir=processed_dir,
        num_classes=num_classes,
        batch_size=batch_size,
        is_training=True,
        use_augmentation=exp.get("use_augmentation", True),
        use_mixup=(p2["mixup_alpha"] > 0 and exp.get("use_augmentation", True)),
        mixup_alpha=p2["mixup_alpha"],
        num_frames=num_frames,
        target_size=(config["frames"]["width"], config["frames"]["height"]),
        seed=42
    )
    val_ds = build_dataset(
        split_csv=val_csv,
        processed_dir=processed_dir,
        num_classes=num_classes,
        batch_size=batch_size,
        is_training=False,
        num_frames=num_frames,
        target_size=(config["frames"]["width"], config["frames"]["height"])
    )

    callbacks_p2 = get_callbacks(
        exp_id=args.exp_id,
        phase="phase2",
        model_dir=model_dir,
        log_dir=log_dir,
        drive_model_dir=args.drive_dir,
        early_stopping_patience=p2["early_stopping_patience"]
    )

    train_phase(
        model, train_ds, val_ds, callbacks_p2,
        max_epochs=p2["max_epochs"],
        phase_name="Phase 2 (Full Fine-Tuning)",
        initial_epoch=initial_epoch_p2,
        class_weight=class_weight_dict
    )

    logger.info(f"\n✅ Experiment {args.exp_id} ({exp['name']}) COMPLETE.")
    logger.info(f"   Best model saved to: {model_dir}/best_model_phase2.keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SSL400 MoViNet model for a specific experiment."
    )
    parser.add_argument(
        "--exp_id",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="Experiment ID: 1=Baseline, 2=CLAHE, 3=Bilateral, 4=Hybrid"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config (use 16 for A100, 8 for T4)"
    )
    parser.add_argument(
        "--drive_dir",
        type=str,
        default=None,
        help="Google Drive directory for checkpoint sync (Colab only)"
    )
    args = parser.parse_args()
    main(args)
