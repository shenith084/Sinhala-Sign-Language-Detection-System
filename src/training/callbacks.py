"""
callbacks.py
============
Phase 2 — Custom Training Callbacks

Defines all Keras callbacks used during training. Callbacks are
experiment-aware: they automatically save to the correct experiment folder.

Callbacks Used:
    1. EarlyStopping      — Stop when val_loss stops improving
    2. ReduceLROnPlateau  — Halve LR when val_loss plateaus
    3. ModelCheckpoint    — Save best model by val_accuracy
    4. CSVLogger          — Log all metrics to CSV each epoch
    5. TensorBoard        — TensorBoard visualization support
    6. TrainingTimer      — Custom: logs epoch duration + ETA
"""

import logging
import time
from pathlib import Path
from typing import List

import tensorflow as tf
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Custom Callback: Training Timer ──────────────────────────────────────────

class TrainingTimer(tf.keras.callbacks.Callback):
    """
    Custom callback that logs:
    - Duration of each epoch in seconds
    - Estimated time remaining for the full training run
    - Current learning rate

    Args:
        total_epochs: Total expected training epochs (for ETA calculation)
    """

    def __init__(self, total_epochs: int = 30):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_start_time = 0.0
        self.training_start_time = 0.0

    def on_train_begin(self, logs=None):
        self.training_start_time = time.time()
        logger.info("Training started.")

    def on_epoch_begin(self, epoch: int, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch: int, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        elapsed_total  = time.time() - self.training_start_time
        epochs_done    = epoch + 1
        epochs_left    = self.total_epochs - epochs_done
        eta_seconds    = (elapsed_total / epochs_done) * epochs_left

        # Get current learning rate
        try:
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        except Exception:
            current_lr = 0.0

        val_acc  = logs.get("val_accuracy", 0.0)
        val_loss = logs.get("val_loss", 0.0)
        train_acc = logs.get("accuracy", 0.0)

        logger.info(
            f"  Epoch {epochs_done:>3}/{self.total_epochs} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
            f"val_loss={val_loss:.4f} | LR={current_lr:.2e} | "
            f"epoch_time={epoch_duration:.1f}s | ETA={eta_seconds/60:.1f}min"
        )


# ── Callback Builder ──────────────────────────────────────────────────────────

def build_callbacks(
    exp_id: int,
    phase: int = 1,
    config_path: str = "config.yaml",
) -> List[tf.keras.callbacks.Callback]:
    """
    Build the full callback list for a given experiment and training phase.

    All file paths are automatically set based on exp_id and phase from config.

    Args:
        exp_id:      Experiment ID (1–5)
        phase:       Training phase (1 = warm-up, 2 = fine-tune)
        config_path: Path to config.yaml

    Returns:
        List of configured Keras callbacks
    """
    config = load_config(config_path)
    cb_cfg = config["callbacks"]
    training_cfg = config["training"]

    exp_key = str(exp_id)
    exp_config = config["experiments"][exp_id]

    model_dir = Path(exp_config["model_dir"])
    log_dir   = Path(exp_config["log_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    max_epochs = (
        training_cfg["max_epochs_phase1"] if phase == 1
        else training_cfg["max_epochs_phase2"]
    )

    # ── 1. EarlyStopping ─────────────────────────────────────────────────────
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=cb_cfg["early_stopping_patience"],
        restore_best_weights=True,
        verbose=1,
        mode="min",
    )

    # ── 2. ReduceLROnPlateau ─────────────────────────────────────────────────
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=cb_cfg["reduce_lr_factor"],
        patience=cb_cfg["reduce_lr_patience"],
        min_lr=cb_cfg["min_lr"],
        verbose=1,
        mode="min",
    )

    # ── 3. ModelCheckpoint ───────────────────────────────────────────────────
    checkpoint_path = model_dir / f"best_model_phase{phase}.keras"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1,
    )

    # ── 4. CSVLogger ─────────────────────────────────────────────────────────
    csv_path = log_dir / f"training_log_phase{phase}.csv"
    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=str(csv_path),
        separator=",",
        append=False,
    )

    # ── 5. TensorBoard ───────────────────────────────────────────────────────
    tb_log_dir = log_dir / f"tensorboard_phase{phase}"
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=str(tb_log_dir),
        histogram_freq=0,
        write_graph=False,
        update_freq="epoch",
    )

    # ── 6. Custom Timer ──────────────────────────────────────────────────────
    timer = TrainingTimer(total_epochs=max_epochs)

    callbacks = [
        early_stopping,
        reduce_lr,
        model_checkpoint,
        csv_logger,
        tensorboard,
        timer,
    ]

    logger.info(f"Callbacks configured for EXP{exp_id} Phase {phase}:")
    logger.info(f"  Checkpoint → {checkpoint_path}")
    logger.info(f"  CSV log    → {csv_path}")
    logger.info(f"  TensorBoard→ {tb_log_dir}")

    return callbacks
