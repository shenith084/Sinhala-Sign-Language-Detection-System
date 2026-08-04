"""
tf_dataset_builder.py
=====================
Builds efficient tf.data.Dataset pipelines from pre-processed .npy files.

Key design decisions:
  - Fast Loading: Uses np.load() to read pre-processed frames, bypassing video decoding.
  - use_augmentation flag controls ALL augmentation (spatial + Mixup):
      EXP1 (Baseline): use_augmentation=False → pure raw baseline, no augmentation at all.
      EXP2-5:          use_augmentation=True  → full pipeline (flip, blur, rotation, Mixup).
  - Augmentation is NEVER applied during validation or testing.
  - Produces one-hot encoded labels.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

AUTOTUNE = tf.data.AUTOTUNE


def load_npy_on_the_fly(
    npy_path: tf.Tensor,
    class_id: tf.Tensor,
    num_classes: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow py_function wrapper: loads an .npy file containing pre-processed frames.
    Returns (frame_tensor, one_hot_label).
    """
    path = npy_path.numpy().decode("utf-8")

    # 1. Load pre-processed numpy array
    tensor = np.load(path)  # shape (num_frames, H, W, 3)

    # 2. One-hot label
    label = tf.one_hot(class_id, num_classes)

    return tf.constant(tensor, dtype=tf.float32), label


def build_dataset(
    split_csv: str,
    processed_dir: str,
    num_classes: int,
    batch_size: int,
    is_training: bool,
    use_augmentation: bool = True,   # EXP1=False (baseline), EXP2-5=True
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    num_frames: int = 32,
    target_size: tuple = (224, 224),
    seed: int = 42
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset pipeline from pre-processed .npy files.

    Args:
        use_augmentation: Controls ALL augmentation (spatial + Mixup).
                          False = EXP1 Baseline (no augmentation at all).
                          True  = EXP2-5 (Flip, Blur, Rotation, Zoom, Mixup active).
        use_mixup:        Only effective when use_augmentation=True.
    """
    df = pd.read_csv(split_csv)
    proc_path = Path(processed_dir)

    npy_paths = []
    class_ids = []

    for _, row in df.iterrows():
        csv_path = Path(str(row["video_path"]))
        video_stem = csv_path.stem
        cid = int(row["class_id"])

        n_path = proc_path / str(cid) / f"{video_stem}.npy"

        if n_path.exists():
            npy_paths.append(str(n_path))
            class_ids.append(cid)

    aug_status = "WITH augmentation" if (is_training and use_augmentation) else "NO augmentation"
    logger.info(
        f"Dataset Pipeline: {len(npy_paths)} NPY files loaded directly, "
        f"{'training' if is_training else 'eval'} mode, {aug_status}."
    )

    if len(npy_paths) == 0:
        logger.warning(f"No .npy files found in {processed_dir}! Did you run video_to_frames.py?")

    path_ds = tf.data.Dataset.from_tensor_slices(npy_paths)
    id_ds = tf.data.Dataset.from_tensor_slices(class_ids)
    ds = tf.data.Dataset.zip((path_ds, id_ds))

    if is_training:
        ds = ds.shuffle(
            buffer_size=len(npy_paths) if len(npy_paths) > 0 else 1,
            seed=seed,
            reshuffle_each_iteration=True
        )

    def _load_sample(path, cid):
        frames, label = tf.py_function(
            func=lambda p, c: load_npy_on_the_fly(p, c, num_classes),
            inp=[path, cid],
            Tout=[tf.float32, tf.float32]
        )
        frames.set_shape([num_frames, target_size[1], target_size[0], 3])
        label.set_shape([num_classes])
        return frames, label

    ds = ds.map(_load_sample, num_parallel_calls=1)

    # -----------------------------------------------------------------------
    # Spatial Augmentation
    # ONLY applied during training AND when use_augmentation=True.
    # EXP1 (Baseline) sets use_augmentation=False → completely skipped.
    # EXP2-5 set use_augmentation=True → Flip, Blur, Rotation, Zoom, etc.
    # -----------------------------------------------------------------------
    if is_training and use_augmentation:
        from training.augmentation import apply_spatial_augmentation

        def _augment(frames, label):
            augmented = tf.py_function(
                func=lambda f: apply_spatial_augmentation(f),
                inp=[frames],
                Tout=tf.float32
            )
            augmented.set_shape([num_frames, target_size[1], target_size[0], 3])
            return augmented, label

        ds = ds.map(_augment, num_parallel_calls=1)

    ds = ds.batch(batch_size, drop_remainder=is_training)

    # -----------------------------------------------------------------------
    # Mixup Augmentation
    # Applied when use_mixup=True, regardless of general augmentation flag.
    # This allows the baseline to use mixup if requested.
    # -----------------------------------------------------------------------
    if is_training and use_mixup:
        from training.augmentation import apply_mixup_batch

        def _mixup(frames_batch, labels_batch):
            m_f, m_l = tf.py_function(
                func=lambda f, l: apply_mixup_batch(f, l, alpha=mixup_alpha),
                inp=[frames_batch, labels_batch],
                Tout=[tf.float32, tf.float32]
            )
            m_f.set_shape([None, num_frames, target_size[1], target_size[0], 3])
            m_l.set_shape([None, num_classes])
            return m_f, m_l

        ds = ds.map(_mixup, num_parallel_calls=1)

    ds = ds.prefetch(AUTOTUNE)

    return ds
