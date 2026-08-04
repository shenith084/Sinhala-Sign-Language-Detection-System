"""
augmentation.py
===============
Spatial and temporal augmentation pipeline for SSL400 training.
Augmentation is ONLY applied during training — never during validation or testing.

Design rationale:
  With ~54 videos/class on average (data reduced, 5 classes), augmentation is critical
  to prevent overfitting and improve generalization to unseen signers.
  Each augmentation simulates a real-world variation in sign language performance.

  Spatial augmentations (ALL ACTIVE):
    - RandomFlip (prob=0.5): Simulates left-handed signers (~10% of population).
      ENABLED after research confirmed all 8 Sinhala signs are flip-safe.
    - RandomBrightness (±10%): Indoor lighting fluctuations between recordings.
    - RandomContrast (±10%): Camera exposure and contrast variation.
    - RandomRotation (±5°): Head/shoulder/camera angle variation.
    - RandomZoom (95–105%): Signer distance from camera variation.
    - GaussianBlur (prob=0.3, kernel 3x3): Camera focus variation & lens quality.
      ADDED: IEEE/MDPI 2024 papers confirm blur augmentation improves robustness
      for sign language models by preventing over-reliance on fine pixel details.

  Temporal augmentations:
    - TemporalJitter: ACTIVE — different signing speeds across signers.
    - SpeedPerturbation (±10%): ENABLED (mild) — time-warping for speed variance.
      Re-enabled with ±10% (was ±20%) now dataset is larger (618 vs old 323).
    - FrameDropout: DISABLED — still too risky on relatively small dataset.

  Mixup (Phase 1 only):
    - ENABLED (alpha=0.2): Blends two videos with random λ from Beta(0.2, 0.2).
    - IEEE/CVPR 2023-24 papers show +3-8% accuracy on small sign language datasets.
    - Forces smooth decision boundaries, prevents memorization of training samples.
    - Phase 2 uses clean gradients only (no Mixup) for precise backbone fine-tuning.
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
    # ENABLED: Research confirmed all 8 Sinhala signs are safe to flip.
    # Flipping simulates left-handed signers (~10% of population).
    # All signs (Thank you, Hello, Good, House, Eat) are either
    # symmetric (House) or dominant-hand based, which remain valid when mirrored.
    if np.random.random() < 0.5:
        frames = frames[:, :, ::-1, :]  # flip all frames horizontally

    # --- Random Brightness (±0.1 in [-1,1] space) ---
    delta = np.random.uniform(-0.1, 0.1)
    frames = np.clip(frames + delta, -1.0, 1.0)
    
    # --- Random Contrast (factor in [0.9, 1.1]) ---
    contrast_factor = np.random.uniform(0.9, 1.1)
    mean = np.mean(frames, axis=(1, 2), keepdims=True)
    frames = np.clip(mean + contrast_factor * (frames - mean), -1.0, 1.0)
    
    # --- Random Rotation (±5°) ---
    angle = np.random.uniform(-5.0, 5.0)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    for i in range(num_frames):
        frames[i] = cv2.warpAffine(frames[i], M, (w, h), borderMode=cv2.BORDER_REFLECT101)
        
    # --- Random Zoom (0.95 – 1.05) ---
    zoom_range = [0.95, 1.05]
    zoom = np.random.uniform(zoom_range[0], zoom_range[1])
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

    # --- Random Gaussian Blur (prob=0.3, kernel 3x3) ---
    # Simulates camera focus variation and lens quality differences between recordings.
    # IEEE/MDPI 2024 SLR papers confirm blur augmentation prevents over-reliance on
    # fine-grained pixel details, forcing the model to learn motion/shape features.
    if np.random.random() < 0.3:
        for i in range(num_frames):
            # Convert [-1,1] → [0,255] for OpenCV blur, then back
            frame_uint8 = ((frames[i] + 1.0) * 127.5).astype(np.uint8)
            blurred = cv2.GaussianBlur(frame_uint8, (3, 3), sigmaX=0.8)
            frames[i] = (blurred.astype(np.float32) / 127.5) - 1.0

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
      head needs to learn smooth decision boundaries across 5 classes.
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
    if alpha <= 0.0:
        return frames_batch, labels_batch

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
