"""
tf_dataset_builder.py
=====================
Builds efficient tf.data.Dataset pipelines directly from raw .mp4 files.

Key design decisions:
  - On-the-fly decoding: Loads raw MP4 files using OpenCV, bypassing disk space limits.
  - Applies enhancement functions dynamically on the CPU.
  - Applies spatial augmentation ONLY during training.
  - Supports Mixup augmentation (Phase 1 only) via a simple flag.
  - Produces one-hot encoded labels (required for label_smoothing + Mixup).
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Callable, List

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AUTOTUNE = tf.data.AUTOTUNE

def read_video_frames(video_path: str) -> List[np.ndarray]:
    """Read all frames from an .mp4 file using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    if len(frames) == 0:
        # Fallback to zero frame
        return [np.zeros((224, 224, 3), dtype=np.uint8)]
    return frames

def uniform_sample_frames(frames: List[np.ndarray], num_frames: int) -> List[np.ndarray]:
    """Uniformly sample exactly `num_frames` from the video."""
    total = len(frames)
    if total >= num_frames:
        indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    else:
        indices = [i % total for i in range(num_frames)]
    return [frames[i] for i in indices]

def load_video_on_the_fly(
    video_path: tf.Tensor,
    class_id: tf.Tensor,
    num_classes: int,
    num_frames: int,
    enhance_fn: Callable[[np.ndarray], np.ndarray],
    target_size: tuple
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow py_function wrapper: loads an MP4 file, samples, enhances, and normalizes.
    Returns (frame_tensor, one_hot_label).
    """
    path = video_path.numpy().decode("utf-8")
    
    # 1. Read
    raw_frames = read_video_frames(path)
    
    # 2. Sample
    sampled = uniform_sample_frames(raw_frames, num_frames)
    
    # 3. Enhance, resize, RGB, normalize
    processed = []
    for frame in sampled:
        if enhance_fn:
            try:
                frame = enhance_fn(frame)
            except Exception:
                pass
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)
        frame = (frame / 127.5) - 1.0
        processed.append(frame)
        
    tensor = np.stack(processed, axis=0) # shape (num_frames, H, W, 3)
    label = tf.one_hot(class_id, num_classes)
    
    return tf.constant(tensor, dtype=tf.float32), label


def build_dataset(
    split_csv: str,
    raw_dir: str,
    num_classes: int,
    batch_size: int,
    is_training: bool,
    enhance_fn: Callable = None,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    num_frames: int = 32,
    target_size: tuple = (224, 224),
    seed: int = 42
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset pipeline from raw MP4 files.
    """
    df = pd.read_csv(split_csv)
    raw_path = Path(raw_dir)

    # Re-map directory structure to find the raw MP4 files
    # The split CSV holds "video_path" as "Category/Class/Video.mp4"
    mp4_paths = []
    class_ids = []
    
    for _, row in df.iterrows():
        # video_path from CSV is like "SSL400/Dataset - Original/Class/Video.mp4"
        # Split it and combine with raw_path
        csv_path = Path(str(row["video_path"]))
        # Extract everything after "Dataset - Original"
        relative_parts = csv_path.parts[csv_path.parts.index("Dataset - Original") + 1:]
        v_path = raw_path.joinpath(*relative_parts)
        
        if v_path.exists():
            mp4_paths.append(str(v_path))
            class_ids.append(int(row["class_id"]))

    logger.info(f"Dataset Pipeline: {len(mp4_paths)} MP4 files loaded directly, "
                f"{'training' if is_training else 'eval'} mode.")

    path_ds = tf.data.Dataset.from_tensor_slices(mp4_paths)
    id_ds = tf.data.Dataset.from_tensor_slices(class_ids)
    ds = tf.data.Dataset.zip((path_ds, id_ds))

    # Shuffle training data
    if is_training:
        ds = ds.shuffle(buffer_size=len(mp4_paths), seed=seed, reshuffle_each_iteration=True)

    # Load MP4 files via py_function
    def _load_sample(path, cid):
        frames, label = tf.py_function(
            func=lambda p, c: load_video_on_the_fly(
                p, c, num_classes, num_frames, enhance_fn, target_size
            ),
            inp=[path, cid],
            Tout=[tf.float32, tf.float32]
        )
        frames.set_shape([num_frames, target_size[1], target_size[0], 3])
        label.set_shape([num_classes])
        return frames, label

    ds = ds.map(_load_sample, num_parallel_calls=1)

    # Apply spatial augmentation during training
    if is_training:
        from training.augmentation import apply_spatial_augmentation
        def _augment(frames, label):
            # Run augmentation in eager mode to avoid symbolic tensor errors
            augmented = tf.py_function(
                func=lambda f: apply_spatial_augmentation(f),
                inp=[frames],
                Tout=tf.float32
            )
            augmented.set_shape([num_frames, target_size[1], target_size[0], 3])
            return augmented, label
        ds = ds.map(_augment, num_parallel_calls=1)

    # Batch
    ds = ds.batch(batch_size, drop_remainder=is_training)

    # Apply Mixup AFTER batching (operates on full batches)
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

    # Prefetch
    ds = ds.prefetch(AUTOTUNE)

    return ds
