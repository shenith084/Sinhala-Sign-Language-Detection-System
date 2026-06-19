"""
clahe_gamma.py
==============
EXP2 — CLAHE + Gamma Correction Enhancement

Addresses: Inconsistent indoor lighting, deep shadows on hands,
           and low-contrast backgrounds that confuse the I3D model.

Pipeline:
    1. Convert BGR → LAB color space
    2. Apply CLAHE to the L (luminance) channel only
       → preserves color, only enhances local contrast
    3. Reconstruct enhanced LAB → BGR
    4. Apply Gamma Correction for global brightness normalization
    5. Resize + Normalize for I3D input

Academic Justification:
    CLAHE is preferred over standard Histogram Equalization (HE) because:
    - HE applies global redistribution → over-brightens highlights, loses
      detail in dark regions.
    - CLAHE uses local adaptive tiling with contrast limiting → enhances
      contrast only where needed, preventing noise amplification.
    - Gamma correction compensates for camera sensor non-linearity.
"""

import numpy as np
import cv2


def enhance_clahe_gamma(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple = (8, 8),
    gamma: float = 1.2,
    target_size: tuple = (224, 224),
) -> np.ndarray:
    """
    Apply CLAHE for local contrast enhancement followed by Gamma Correction
    for global brightness normalization, then normalize for I3D input.

    Args:
        frame:       BGR frame as numpy uint8 array (H, W, 3)
        clip_limit:  CLAHE contrast limit. 2.0 = moderate enhancement.
                     Higher values increase contrast but risk noise.
        tile_grid:   CLAHE tile grid size (rows, cols). (8,8) works well
                     for 224×224 resolution.
        gamma:       Gamma value. > 1.0 brightens (corrects dark frames).
                     < 1.0 darkens. 1.2 is a gentle brightening.
        target_size: Output spatial size (width, height). Default (224, 224)

    Returns:
        Enhanced, normalized float32 frame in range [-1, 1], shape (224, 224, 3) RGB

    Raises:
        ValueError: If frame is None or empty
    """
    if frame is None or frame.size == 0:
        raise ValueError("enhance_clahe_gamma received an empty or None frame.")

    # ── Step 1: Convert BGR → LAB ─────────────────────────────────────────────
    # LAB separates luminance (L) from color (A, B).
    # Applying CLAHE only to L avoids color distortion.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # ── Step 2: Apply CLAHE to L channel ─────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_channel)

    # ── Step 3: Merge enhanced L back and convert to BGR ─────────────────────
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    frame_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # ── Step 4: Gamma Correction ──────────────────────────────────────────────
    # Build a lookup table for fast per-pixel gamma application.
    # Formula: output = (input / 255) ^ (1/gamma) * 255
    inv_gamma = 1.0 / gamma
    gamma_table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in range(256)],
        dtype=np.uint8,
    )
    frame_gamma = cv2.LUT(frame_enhanced, gamma_table)

    # ── Step 5: Resize to I3D input size ─────────────────────────────────────
    frame_resized = cv2.resize(frame_gamma, target_size, interpolation=cv2.INTER_LINEAR)

    # ── Step 6: Convert BGR → RGB + Normalize to [-1, 1] ─────────────────────
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_float = frame_rgb.astype(np.float32)
    frame_normalized = (frame_float / 127.5) - 1.0

    return frame_normalized
