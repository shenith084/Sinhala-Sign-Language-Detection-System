"""
callbacks.py
============
Custom Keras training callbacks for the SSL400 research project.

Provides:
  - ProgressLogger: Epoch-level progress logger with tqdm
  - GoogleDriveSync: Syncs checkpoint to Google Drive after each epoch (Colab)
  - ResumeCheckpointFinder: Detects existing checkpoints for resume logic
"""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import tf_keras as keras
except ImportError:
    import tensorflow.keras as keras

logger = logging.getLogger(__name__)


class ProgressLogger(keras.callbacks.Callback):
    """
    Logs epoch-level metrics to both console and a running in-memory list.
    Compatible with both tqdm and standard logging.

    Args:
        exp_id:    Experiment identifier (for labeling log messages).
        phase:     Training phase ('phase1' or 'phase2').
    """

    def __init__(self, exp_id: int, phase: str = "phase1"):
        super().__init__()
        self.exp_id = exp_id
        self.phase = phase
        self.epoch_start_time = 0.0

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        """Record epoch start time."""
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Log all metrics at the end of each epoch."""
        logs = logs or {}
        elapsed = time.time() - self.epoch_start_time
        metrics_str = "  ".join(
            f"{k}: {v:.4f}" for k, v in sorted(logs.items())
        )
        logger.info(
            f"[EXP{self.exp_id} {self.phase.upper()}] "
            f"Epoch {epoch + 1:03d} | {elapsed:.0f}s | {metrics_str}"
        )


class GoogleDriveSync(keras.callbacks.Callback):
    """
    Syncs the model checkpoint directory to Google Drive after every epoch.
    Only activates when running inside Google Colab.

    This is the key resume mechanism: if the Colab session disconnects,
    the latest checkpoint is already saved to Drive and can be resumed
    immediately on the next session without losing progress.

    Args:
        local_model_dir:  Local path to the experiment's model directory.
        drive_model_dir:  Google Drive destination path.
        local_log_dir:    Local path to the experiment's log directory.
        sync_every_n:     Sync frequency in epochs (default: 1 = every epoch).
    """

    def __init__(
        self,
        local_model_dir: str,
        drive_model_dir: str,
        local_log_dir: Optional[str] = None,
        sync_every_n: int = 1
    ):
        super().__init__()
        self.local_dir = Path(local_model_dir)
        self.drive_dir = Path(drive_model_dir)
        self.local_log_dir = Path(local_log_dir) if local_log_dir else None
        self.sync_every_n = sync_every_n
        self._is_colab = self._check_colab()

    def _check_colab(self) -> bool:
        """Detect if running in Google Colab."""
        try:
            import google.colab  # type: ignore
            return True
        except ImportError:
            return False

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        """Sync to Google Drive at the specified frequency."""
        if not self._is_colab:
            return
        if (epoch + 1) % self.sync_every_n != 0:
            return

        try:
            self.drive_dir.mkdir(parents=True, exist_ok=True)
            # Sync model files
            for f in self.local_dir.iterdir():
                if f.is_file() and f.suffix in [".keras", ".h5", ".csv", ".txt"]:
                    shutil.copy2(str(f), str(self.drive_dir / f.name))
                    
            # Sync log files
            if self.local_log_dir and self.local_log_dir.exists():
                for f in self.local_log_dir.iterdir():
                    if f.is_file() and f.suffix in [".csv", ".txt"]:
                        shutil.copy2(str(f), str(self.drive_dir / f.name))
                        
            logger.info(f"[Drive Sync] Epoch {epoch + 1}: synced → {self.drive_dir}")
        except Exception as e:
            logger.warning(f"[Drive Sync] Failed at epoch {epoch + 1}: {e}")


def get_callbacks(
    exp_id: int,
    phase: str,
    model_dir: str,
    log_dir: str,
    drive_model_dir: Optional[str] = None,
    early_stopping_patience: int = 15
) -> list:
    """
    Build and return the full callback list for a training phase.

    Includes:
    1. EarlyStopping: Stops training when val_loss stops improving.
    2. ModelCheckpoint: Saves the best model (lowest val_loss).
    3. CSVLogger: Appends per-epoch metrics to a CSV file.
    4. ProgressLogger: Human-readable epoch progress.
    5. GoogleDriveSync: Colab resume support (only in Colab).

    Args:
        exp_id:                   Experiment ID (1–5).
        phase:                    'phase1' or 'phase2'.
        model_dir:                Local directory to save model checkpoints.
        log_dir:                  Local directory to save CSV logs.
        drive_model_dir:          Google Drive sync destination (None = skip).
        early_stopping_patience:  Epochs without improvement before stopping.

    Returns:
        List of configured Keras Callback objects.
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_path = str(Path(model_dir) / f"best_model_{phase}.keras")
    log_path = str(Path(log_dir) / f"training_log_{phase}.csv")

    callbacks = [
        # 1. Early Stopping
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),

        # 2. Model Checkpoint — saves best val_loss model
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            save_weights_only=False,
            verbose=1
        ),

        # 3. CSV Logger — append mode so resume works correctly
        keras.callbacks.CSVLogger(
            filename=log_path,
            separator=",",
            append=True
        ),

        # 4. Human-readable progress logger
        ProgressLogger(exp_id=exp_id, phase=phase)
    ]

    # 6. Google Drive sync (Colab only)
    if drive_model_dir:
        callbacks.append(
            GoogleDriveSync(
                local_model_dir=model_dir,
                drive_model_dir=drive_model_dir,
                local_log_dir=log_dir,
                sync_every_n=1
            )
        )

    return callbacks


def find_resume_checkpoint(model_dir: str, phase: str) -> Optional[str]:
    """
    Check if a checkpoint exists for the given experiment and phase.

    Used at the start of training to decide whether to load weights
    and resume from where a previous session left off.

    Args:
        model_dir: Local model directory (e.g., 'models/experiment_1').
        phase:     'phase1' or 'phase2'.

    Returns:
        Path to the checkpoint file if it exists, else None.
    """
    checkpoint_path = Path(model_dir) / f"best_model_{phase}.keras"
    if checkpoint_path.exists():
        logger.info(f"[Resume] Found checkpoint: {checkpoint_path}")
        return str(checkpoint_path)
    logger.info(f"[Resume] No checkpoint found for {phase} in {model_dir}. "
                "Starting fresh.")
    return None
