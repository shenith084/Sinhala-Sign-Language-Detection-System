"""
bilateral.py — Experiment 3: Bilateral Filter (Edge-Preserving Denoising)
==========================================================================
Addresses background motion noise and skin-texture noise in SSL400 videos
while preserving the precise contours of hand shapes critical for gesture recognition.

WHY Bilateral Filter over Gaussian Blur:
  Gaussian blur is spatially uniform — it smooths everything equally, including
  the sharp edges at finger boundaries and joint contours. This actively DESTROYS
  the high-frequency spatial features that MoViNet's convolutional layers rely on.

  The Bilateral Filter is edge-aware: it computes smoothing weights based on
  BOTH spatial proximity AND intensity similarity. Pixels with very different
  intensity values (i.e., across an edge) receive near-zero smoothing weights,
  preserving the boundary. Homogeneous regions (background, sky, walls) are
  strongly smoothed, removing noise without touching hand contours.

  This makes it ideal for sign language where hand shape boundaries are the
  primary discriminative features.
"""

import cv2
import numpy as np


def enhance_bilateral(
    frame: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    """
    Apply Bilateral Filter for edge-preserving noise reduction.

    The bilateral filter smooths homogeneous regions (background noise)
    while strictly preserving high-contrast edges (finger contours, joint angles).

    Args:
        frame:       BGR uint8 numpy array (OpenCV frame).
        d:           Diameter of each pixel's neighborhood.
                     9 = strong denoising; 5 = mild. Higher values are slower.
        sigma_color: Filter sigma in color space. Larger values mean more distant
                     colors are included in the smoothing. 75 = moderate blending.
        sigma_space: Filter sigma in coordinate space. Larger values mean pixels
                     further away influence the smoothing. 75 = ~9px effective radius.

    Returns:
        Denoised BGR uint8 frame with sharp hand edges preserved.
    """
    filtered = cv2.bilateralFilter(
        frame,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )
    return filtered
