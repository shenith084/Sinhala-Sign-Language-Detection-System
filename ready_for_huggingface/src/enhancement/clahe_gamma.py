"""
clahe_gamma.py — Experiment 2: CLAHE + Gamma Correction
=========================================================
Addresses the most common failure mode in SSL400: inconsistent indoor lighting,
deep shadows on hands, and low-contrast backgrounds.

WHY CLAHE over standard Histogram Equalization:
  Standard HE amplifies noise uniformly across the entire image.
  CLAHE uses a local tile grid (8×8) with a contrast limit to enhance
  contrast ONLY where needed, preventing over-brightening and noise amplification.
  Operating on the LAB L-channel preserves natural skin tones and color fidelity.

WHY Gamma Correction after CLAHE:
  CLAHE fixes local contrast but can still leave global brightness uneven.
  Gamma correction (γ=1.2) brightens midtones globally, improving visibility
  of hand regions filmed in dim indoor environments.
"""

import cv2
import numpy as np


def enhance_clahe_gamma(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple = (8, 8),
    gamma: float = 1.2
) -> np.ndarray:
    """
    Apply CLAHE for local contrast enhancement followed by Gamma Correction
    for global brightness normalization.

    Pipeline:
        BGR → LAB → CLAHE(L-channel) → LAB → BGR → Gamma Correction → BGR

    Args:
        frame:      BGR uint8 numpy array (OpenCV frame).
        clip_limit: CLAHE contrast limit. 2.0 = moderate, balanced enhancement.
                    Higher values increase contrast but risk noise amplification.
        tile_grid:  CLAHE tile grid size. (8, 8) works well for 224×224 frames.
        gamma:      Gamma value. > 1.0 brightens, < 1.0 darkens.
                    1.2 provides gentle brightening without over-exposing.

    Returns:
        Enhanced BGR uint8 frame with improved local contrast and brightness.

    Raises:
        cv2.error: If the input frame is not a valid uint8 BGR image.
    """
    # Step 1: Convert BGR → LAB color space
    # Reason: CLAHE applied to LAB L-channel enhances luminance only,
    # preserving the natural color of skin and clothing.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Step 2: Apply CLAHE to the L (luminance) channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_channel)

    # Step 3: Merge back and convert to BGR
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Step 4: Apply Gamma Correction using a pre-computed LUT (fast)
    inv_gamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in range(256)],
        dtype=np.uint8
    )
    enhanced = cv2.LUT(enhanced, table)

    return enhanced
