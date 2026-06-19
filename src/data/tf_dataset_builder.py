"""
tf_dataset_builder.py
=====================
Phase 1, Step 1.3 — TensorFlow Data Pipeline Builder

Builds tf.data.Dataset pipelines for train/val/test splits.
Each dataset:
  1. Reads a split CSV (list of video paths + class IDs)
  2. Loads each .mov/.mp4 video with OpenCV
  3. Uniformly samples `target_frames` frames from each clip
  4. Applies the experiment's enhancement function per frame
  5. Returns (video_tensor, one_hot_label) pairs

Design Decisions:
  - Python generator wrapped in tf.data.Dataset.from_generator():
    Avoids loading all videos into memory. Each video is decoded lazily.
  - Enhancement is applied INSIDE the generator (CPU preprocessing).
  - tf.data pipeline uses prefetch() + num_parallel_calls for efficiency.
  - Augmentation is applied as a tf.data .map() step (train only).
  - One-hot encoding for labels (CategoricalCrossentropy loss).

Usage:
    from src.data.tf_dataset_builder import build_dataset

    train_ds = build_dataset(
        split_csv="data/splits/train_split.csv",
        exp_id=2,
        batch_size=4,
        num_classes=383,
        augment=True,
    )
"""

import logging
import os
from pathlib import Path
from typing import Callable, Generator, Tuple

import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import yaml

# ── Suppress TF info logs ─────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Core Video Loader ─────────────────────────────────────────────────────────

def load_video_frames(
    video_path: str,
    target_frames: int = 32,
    enhance_fn: Callable = None,
    temporal_jitter: bool = False,
) -> np.ndarray:
    """
    Load a video file and return a uniformly sampled, enhanced frame stack.

    Args:
        video_path:     Absolute or relative path to the video file
        target_frames:  Number of frames to sample from the clip
        enhance_fn:     Enhancement callable: (BGR uint8) → (RGB float32 [-1,1])
                        If None, applies baseline resize+normalize only.
        temporal_jitter: If True, uses random frame sampling instead of uniform.
                         Only enabled during training for temporal augmentation.

    Returns:
        Float32 numpy array of shape (target_frames, 224, 224, 3)
        Pixel values in range [-1, 1]

    Raises:
        IOError: If video cannot be opened or has insufficient frames
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Read all frames from the video
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)

    if total_frames == 0:
        raise IOError(f"No frames decoded from video: {video_path}")

    # ── Frame Sampling ────────────────────────────────────────────────────────
    if total_frames <= target_frames:
        # Pad by repeating last frame if clip is shorter than target
        indices = list(range(total_frames))
        while len(indices) < target_frames:
            indices.append(indices[-1])
    elif temporal_jitter:
        # Temporal augmentation: random sampling
        indices = sorted(
            np.random.choice(total_frames, size=target_frames, replace=False)
        )
    else:
        # Uniform sampling: evenly spaced indices across clip
        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int).tolist()

    # ── Apply Enhancement Per Frame ───────────────────────────────────────────
    processed_frames = []
    for idx in indices:
        frame_bgr = frames[idx]

        if enhance_fn is not None:
            try:
                frame_processed = enhance_fn(frame_bgr)
            except Exception as e:
                logger.warning(f"Enhancement failed on frame {idx} of {video_path}: {e}")
                # Fall back to baseline processing
                frame_rgb = cv2.cvtColor(
                    cv2.resize(frame_bgr, (224, 224)), cv2.COLOR_BGR2RGB
                )
                frame_processed = (frame_rgb.astype(np.float32) / 127.5) - 1.0
        else:
            # No enhancement: baseline resize + normalize
            frame_rgb = cv2.cvtColor(
                cv2.resize(frame_bgr, (224, 224)), cv2.COLOR_BGR2RGB
            )
            frame_processed = (frame_rgb.astype(np.float32) / 127.5) - 1.0

        processed_frames.append(frame_processed)

    # Stack into (T, H, W, C) tensor
    video_tensor = np.stack(processed_frames, axis=0).astype(np.float32)
    return video_tensor


# ── Generator ─────────────────────────────────────────────────────────────────

def video_generator(
    split_csv: str,
    enhance_fn: Callable,
    num_classes: int,
    target_frames: int = 32,
    temporal_jitter: bool = False,
    shuffle: bool = False,
    seed: int = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Python generator that yields (video_tensor, one_hot_label) pairs.

    Args:
        split_csv:       Path to split CSV (train/val/test)
        enhance_fn:      Enhancement function from enhancement_factory
        num_classes:     Total number of classes (383 for SSL400)
        target_frames:   Frames per clip (default 32)
        temporal_jitter: Enable random frame sampling (training only)
        shuffle:         Shuffle rows before iteration (training only)
        seed:            Random seed for shuffle/jitter

    Yields:
        Tuple of:
          - video_tensor: float32 array (target_frames, 224, 224, 3)
          - one_hot:      float32 array (num_classes,)
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(split_csv)

    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    for _, row in df.iterrows():
        video_path = row["video_path"].replace("\\", "/")
        class_id   = int(row["class_id"])

        try:
            video_tensor = load_video_frames(
                video_path=video_path,
                target_frames=target_frames,
                enhance_fn=enhance_fn,
                temporal_jitter=temporal_jitter,
            )
        except (IOError, Exception) as e:
            logger.warning(f"Skipping {video_path}: {e}")
            continue

        # One-hot encode label
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[class_id] = 1.0

        yield video_tensor, one_hot


# ── Dataset Builder ───────────────────────────────────────────────────────────

def build_dataset(
    split_csv: str,
    exp_id: int,
    batch_size: int = 4,
    num_classes: int = 383,
    target_frames: int = 32,
    augment: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    config_path: str = "config.yaml",
    prefetch: bool = True,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset pipeline for a given split and experiment.

    Args:
        split_csv:     Path to the split CSV file
        exp_id:        Experiment ID (1–5) — controls which enhancement is applied
        batch_size:    Batch size for training (4 for CPU, 8 for GPU)
        num_classes:   Number of output classes (383 for SSL400)
        target_frames: Frames per video clip (32 for I3D standard)
        augment:       Whether to apply spatial augmentation (train only)
        shuffle:       Whether to shuffle the dataset (train only)
        seed:          Random seed for reproducibility
        config_path:   Path to config.yaml
        prefetch:      Whether to prefetch batches for performance

    Returns:
        Batched tf.data.Dataset yielding (video_batch, label_batch) tensors
        video_batch shape: (batch_size, target_frames, 224, 224, 3)
        label_batch shape: (batch_size, num_classes)
    """
    # Import here to avoid circular imports
    from src.enhancement.enhancement_factory import get_enhancer

    enhance_fn = get_enhancer(exp_id=exp_id, config_path=config_path)
    temporal_jitter = augment   # Use jitter only during training

    # Define output signature for tf.data.Dataset.from_generator
    output_signature = (
        tf.TensorSpec(shape=(target_frames, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(num_classes,),               dtype=tf.float32),
    )

    # Wrap Python generator
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: video_generator(
            split_csv=split_csv,
            enhance_fn=enhance_fn,
            num_classes=num_classes,
            target_frames=target_frames,
            temporal_jitter=temporal_jitter,
            shuffle=shuffle,
            seed=seed,
        ),
        output_signature=output_signature,
    )

    # ── Spatial Augmentation (training only) ──────────────────────────────────
    if augment:
        dataset = dataset.map(
            _apply_spatial_augmentation,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # ── Batch → Prefetch ──────────────────────────────────────────────────────
    dataset = dataset.batch(batch_size, drop_remainder=False)

    if prefetch:
        # User has Colab Pro with 167GB System RAM + A100 GPU
        # Re-enabling AUTOTUNE to maximize data throughput and prevent GPU starvation
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ── Spatial Augmentation ──────────────────────────────────────────────────────

@tf.function
def _apply_spatial_augmentation(
    video: tf.Tensor,
    label: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply spatial augmentation to a single video clip (all frames consistently).

    All augmentation is applied CONSISTENTLY across ALL frames in the clip.
    This is critical: if we flip frame 1 but not frame 2, we destroy temporal
    continuity and confuse the 3D convolutions.

    Augmentations applied:
      - Random horizontal flip (prob=0.5) — simulates left-handed signers
      - Random brightness (±0.2)          — lighting variation
      - Random contrast (0.8–1.2)         — camera exposure variation

    Note: Rotation and zoom are omitted here because applying them
    consistently across 32 frames in TF graph mode is expensive.
    These are best handled in a CPU preprocessing step if needed.

    Args:
        video: float32 tensor (T, H, W, 3) in range [-1, 1]
        label: float32 one-hot tensor (num_classes,)

    Returns:
        Augmented (video, label) tuple
    """
    # Random horizontal flip — same decision for ALL frames
    flip = tf.random.uniform(()) > 0.5
    video = tf.cond(flip, lambda: tf.image.flip_left_right(video), lambda: video)

    # Random brightness — same delta for ALL frames (consistent lighting)
    brightness_delta = tf.random.uniform((), -0.2, 0.2)

    def apply_brightness(v):
        v = v + brightness_delta
        return tf.clip_by_value(v, -1.0, 1.0)

    video = apply_brightness(video)

    # Random contrast — apply per-frame (contrast doesn't break temporal flow)
    # Shift to [0,1] for contrast adjustment, then back to [-1,1]
    video_01 = (video + 1.0) / 2.0
    contrast_factor = tf.random.uniform((), 0.8, 1.2)
    mean = tf.reduce_mean(video_01, axis=[1, 2], keepdims=True)
    video_01 = tf.clip_by_value((video_01 - mean) * contrast_factor + mean, 0.0, 1.0)
    video = video_01 * 2.0 - 1.0

    return video, label


# ── Quick Test ────────────────────────────────────────────────────────────────

def test_pipeline(exp_id: int = 1, n_samples: int = 2) -> None:
    """
    Quick sanity check: load n_samples from the training split and print shapes.

    Args:
        exp_id:    Experiment ID to test
        n_samples: Number of samples to check
    """
    config = load_config()
    num_classes = config["model"]["num_classes"]
    target_frames = config["video"]["target_frames"]

    split_csv = "data/splits/train_split.csv"

    logger.info(f"Testing pipeline for Experiment {exp_id}...")
    dataset = build_dataset(
        split_csv=split_csv,
        exp_id=exp_id,
        batch_size=1,
        num_classes=num_classes,
        target_frames=target_frames,
        augment=False,
        shuffle=False,
        prefetch=False,
    )

    for i, (video, label) in enumerate(dataset.take(n_samples)):
        logger.info(
            f"  Sample {i+1}: video={video.shape}, label={label.shape}, "
            f"class_id={tf.argmax(label[0]).numpy()}, "
            f"pixel_range=[{video.numpy().min():.2f}, {video.numpy().max():.2f}]"
        )

    logger.info("  ✅ Pipeline test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    test_pipeline(exp_id=1, n_samples=2)
