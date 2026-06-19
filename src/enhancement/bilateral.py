"""
bilateral.py
============
EXP3 — Bilateral Filter (Edge-Preserving Noise Reduction)

Addresses: Background motion noise, skin-texture noise, and compression
           artifacts that create false edges confusing I3D spatial filters.

Pipeline:
    1. Apply Bilateral Filter for edge-preserving smoothing
    2. Resize + Normalize for I3D input

Academic Justification:
    Bilateral Filter is preferred over Gaussian Blur because:
    - Gaussian Blur: indiscriminate smoothing → blurs finger edges and
      hand contours that I3D relies on for spatial feature extraction.
    - Bilateral Filter: edge-aware → smooths regions of similar intensity
      (flat background, skin tone regions) while preserving high-contrast
      boundaries (finger contours, hand edges, knuckle details).
    - The filter simultaneously considers spatial closeness (sigma_space)
      and color similarity (sigma_color), making it ideal for noisy video.
"""

import numpy as np
import cv2


def enhance_bilateral(
    frame: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    target_size: tuple = (224, 224),
) -> np.ndarray:
    """
    Apply Bilateral Filter for edge-preserving noise reduction,
    then normalize for I3D input.

    Args:
        frame:        BGR frame as numpy uint8 array (H, W, 3)
        d:            Diameter of each pixel neighborhood.
                      9 = strong filtering (good for noisy video).
                      5 = mild filtering (preserves more texture).
        sigma_color:  Filter sigma in color space. Larger values mean
                      pixels with more dissimilar colors are blended.
                      75 = moderate (blends similar skin/background tones).
        sigma_space:  Filter sigma in coordinate space. Larger values mean
                      pixels farther away influence each other.
                      75 = medium spatial radius.
        target_size:  Output spatial size (width, height). Default (224, 224)

    Returns:
        Denoised, normalized float32 frame in range [-1, 1], shape (224, 224, 3) RGB

    Raises:
        ValueError: If frame is None or empty
    """
    if frame is None or frame.size == 0:
        raise ValueError("enhance_bilateral received an empty or None frame.")

    # ── Step 1: Apply Bilateral Filter ───────────────────────────────────────
    # Note: Bilateral filter is computationally expensive with large d.
    # d=9 is a practical balance between quality and speed for video.
    frame_filtered = cv2.bilateralFilter(
        frame,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    # ── Step 2: Resize to I3D input size ─────────────────────────────────────
    frame_resized = cv2.resize(frame_filtered, target_size, interpolation=cv2.INTER_LINEAR)

    # ── Step 3: Convert BGR → RGB + Normalize to [-1, 1] ─────────────────────
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_float = frame_rgb.astype(np.float32)
    frame_normalized = (frame_float / 127.5) - 1.0

    return frame_normalized
