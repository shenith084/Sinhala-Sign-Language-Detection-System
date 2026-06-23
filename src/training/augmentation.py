"""
augmentation.py
===============
Spatial and temporal augmentation pipeline for SSL400 training.
Augmentation is ONLY applied during training — never during validation or testing.

Design rationale:
  With ~21 videos/class on average (minimum 8), the model will catastrophically
  overfit without aggressive augmentation. Each augmentation simulates a real-world
  variation in how sign language is performed:

  Spatial augmentations:
    - RandomFlip: Simulates left-handed signers (≈10% of population)
    - RandomRotation: Head/shoulder angle variation when filming
    - RandomZoom: Distance from camera variation
    - RandomBrightness/Contrast: Indoor lighting fluctuations

  Temporal augmentations:
    - TemporalJitter: Different signers perform at different speeds
    - SpeedPerturbation: Stretch/compress clip by ±20% then re-sample
    - FrameDropout: Simulates encoding artifacts / dropped frames

  Mixup (Phase 1 only):
    - Blends two videos and their labels with a random λ from Beta(0.2, 0.2)
    - Forces the model to learn continuous transitions between gestures
    - Mathematically prevents memorization of individual training samples
"""

import os
import random
from typing import Tuple

import numpy as np
import tensorflow as tf

# Fix seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------------------------
# Spatial Augmentation
# ---------------------------------------------------------------------------

def apply_spatial_augmentation(
    frames,
    seed: int = SEED
) -> np.ndarray:
    """Apply spatial augmentations using pure NumPy and OpenCV."""
    import cv2
    import numpy as np
    
    if hasattr(frames, 'numpy'):
        frames = frames.numpy()
        
    num_frames, h, w, c = frames.shape
    
    # --- Random Horizontal Flip (prob=0.5) ---
    if np.random.rand() > 0.5:
        frames = np.flip(frames, axis=2)
        
    # --- Random Brightness (±0.2 in [-1,1] space) ---
    delta = np.random.uniform(-0.2, 0.2)
    frames = np.clip(frames + delta, -1.0, 1.0)
    
    # --- Random Contrast (factor in [0.8, 1.2]) ---
    contrast_factor = np.random.uniform(0.8, 1.2)
    mean = np.mean(frames, axis=(1, 2), keepdims=True)
    frames = np.clip(mean + contrast_factor * (frames - mean), -1.0, 1.0)
    
    # --- Random Rotation (±15°) ---
    angle = np.random.uniform(-15.0, 15.0)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    for i in range(num_frames):
        frames[i] = cv2.warpAffine(frames[i], M, (w, h), borderMode=cv2.BORDER_REFLECT101)
        
    # --- Random Zoom (0.85 – 1.15) ---
    zoom = np.random.uniform(0.85, 1.15)
    if zoom < 1.0:
        pad_h = int(h * (1.0 - zoom) / 2)
        pad_w = int(w * (1.0 - zoom) / 2)
        padded = np.pad(frames, ((0,0), (pad_h, pad_h), (pad_w, pad_w), (0,0)), mode='edge')
        for i in range(num_frames):
            frames[i] = cv2.resize(padded[i], (w, h))
    else:
        crop_h = int(h * (zoom - 1.0) / 2)
        crop_w = int(w * (zoom - 1.0) / 2)
        crop_h = max(0, min(crop_h, h - 1))
        crop_w = max(0, min(crop_w, w - 1))
        cropped = frames[:, crop_h:h-crop_h, crop_w:w-crop_w, :]
        for i in range(num_frames):
            frames[i] = cv2.resize(cropped[i], (w, h))
            
    return frames.astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal Augmentation
# ---------------------------------------------------------------------------

def apply_temporal_jitter(
    frames: np.ndarray,
    num_target_frames: int = 32
) -> np.ndarray:
    """
    Randomly sample `num_target_frames` frames instead of uniform sampling.

    Simulates different signing speeds: faster signers complete a gesture in
    fewer frames; slower signers spread it across more frames.

    Args:
        frames:            NumPy array of all decoded frames from a video, shape (N, H, W, C).
        num_target_frames: Number of frames to output.

    Returns:
        NumPy array of shape (num_target_frames, H, W, C) with jittered sampling.
    """
    total = len(frames)
    if total <= num_target_frames:
        # Repeat frames if video is short
        indices = [i % total for i in range(num_target_frames)]
    else:
        # Random start point for the sampling window
        max_start = total - num_target_frames
        start = random.randint(0, max_start)
        # Uniform sample within the jittered window
        indices = np.linspace(start, min(start + num_target_frames * 2 - 1, total - 1),
                              num=num_target_frames, dtype=int)

    return np.stack([frames[i] for i in indices], axis=0)


def apply_speed_perturbation(
    frames: np.ndarray,
    num_target_frames: int = 32,
    speed_range: Tuple[float, float] = (0.8, 1.2)
) -> np.ndarray:
    """
    Stretch or compress a clip by a random speed factor, then resample.

    A speed of 0.8 means the gesture appears 20% slower (more frames sampled from
    a smaller temporal window). 1.2 means 20% faster (fewer frames in the window).

    Args:
        frames:            All decoded frames, shape (N, H, W, C).
        num_target_frames: Target frame count after perturbation.
        speed_range:       (min_speed, max_speed) perturbation factor.

    Returns:
        Resampled tensor of shape (num_target_frames, H, W, C).
    """
    total = len(frames)
    speed = random.uniform(*speed_range)

    # Effective window size after speed perturbation
    effective_length = int(total / speed)
    effective_length = max(num_target_frames, min(effective_length, total))

    # Sample from the effective window
    indices = np.linspace(0, effective_length - 1, num=num_target_frames, dtype=int)
    indices = np.clip(indices, 0, total - 1)

    return np.stack([frames[i] for i in indices], axis=0)


def apply_frame_dropout(
    frames: np.ndarray,
    max_drop: int = 2
) -> np.ndarray:
    """
    Randomly drop 1–max_drop frames and duplicate their neighbors.

    Simulates video encoding artifacts (dropped frames from compression),
    improving robustness to real-world video quality issues.

    Args:
        frames:   NumPy array, shape (num_frames, H, W, C).
        max_drop: Maximum number of frames to drop and duplicate.

    Returns:
        NumPy array of same shape (num_frames, H, W, C) with some frames duplicated.
    """
    n = len(frames)
    num_drop = random.randint(1, min(max_drop, n - 1))
    drop_indices = sorted(random.sample(range(1, n - 1), num_drop))

    frames_list = list(frames)
    for i in drop_indices:
        # Replace dropped frame with previous frame (neighbor duplication)
        frames_list[i] = frames_list[i - 1].copy()

    return np.stack(frames_list, axis=0)


# ---------------------------------------------------------------------------
# Mixup Augmentation
# ---------------------------------------------------------------------------

def apply_mixup_batch(
    frames_batch: tf.Tensor,
    labels_batch: tf.Tensor,
    alpha: float = 0.2
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply Mixup augmentation to a batch of video tensors and one-hot labels.

    Mixup blends two samples linearly:
        mixed_x = λ * x_i + (1 - λ) * x_j
        mixed_y = λ * y_i + (1 - λ) * y_j
    where λ ~ Beta(alpha, alpha)

    Why Mixup in Phase 1 only:
    - During head warm-up, the backbone extracts general features and the new
      head needs to learn smooth decision boundaries across 150 classes.
    - Mixup forces the loss surface to be convex between any two classes,
      dramatically reducing overfitting on 8–35 samples/class.
    - In Phase 2 (full fine-tuning), clean gradients are needed to precisely
      adapt the backbone's weights, so Mixup is removed.

    Args:
        frames_batch: Float32 tensor, shape (batch, num_frames, H, W, C).
        labels_batch: Float32 tensor of one-hot labels, shape (batch, num_classes).
        alpha:        Beta distribution parameter. 0.2 = gentle mixing.

    Returns:
        Tuple of (mixed_frames, mixed_labels) tensors of the same shapes.
    """
    batch_size = tf.shape(frames_batch)[0]

    # Sample λ from Beta(alpha, alpha)
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)  # Use the larger of λ, 1-λ to avoid near-50/50 mixes

    # Shuffle indices for pairing
    indices = tf.random.shuffle(tf.range(batch_size), seed=SEED)

    # Mix frames and labels
    mixed_frames = lam * frames_batch + (1.0 - lam) * tf.gather(frames_batch, indices)
    mixed_labels = lam * labels_batch + (1.0 - lam) * tf.gather(labels_batch, indices)

    return mixed_frames, mixed_labels
