"""
unsharp.py — Experiment 4: Unsharp Masking (Detail Enhancement / Sharpening)
==============================================================================
Amplifies fine-grained spatial details that are critical for gesture discrimination:
finger separation, joint angles, palm texture, and hand-shape boundaries.

WHY Unsharp Masking works:
  The technique creates a "sharpening mask" by subtracting a Gaussian-blurred
  version of the image from the original. This mask captures edges and details.
  Adding a weighted version of this mask back to the original amplifies those details.

  Formula: sharpened = original + amount × (original − blurred)

  For sign language, this is academically motivated by its use in medical imaging
  (e.g., X-ray and MRI enhancement) to improve feature detectability. The same
  principle applies: our model's convolutional filters will detect more discriminative
  spatial features in sharpened frames.

  Note: `threshold=0` applies sharpening everywhere. A nonzero threshold would
  restrict sharpening to areas with strong edges, avoiding noise amplification in
  smooth regions.
"""

import cv2
import numpy as np


def enhance_unsharp_masking(
    frame: np.ndarray,
    kernel_size: tuple = (5, 5),
    sigma: float = 1.0,
    amount: float = 1.5,
    threshold: int = 0
) -> np.ndarray:
    """
    Apply Unsharp Masking to enhance fine spatial details in a video frame.

    Mathematical formula:
        sharpened = clip(original + amount × (original − blurred), 0, 255)

    Args:
        frame:       BGR uint8 numpy array (OpenCV frame).
        kernel_size: Gaussian blur kernel size for creating the unsharp mask.
                     (5, 5) provides a moderate scale of detail enhancement.
        sigma:       Gaussian blur sigma. 1.0 = standard deviation of 1 pixel.
        amount:      Sharpening strength. 1.5 = aggressive but controlled.
                     Lower values (0.5–1.0) are more subtle.
        threshold:   Minimum pixel difference to apply sharpening.
                     0 = apply everywhere; higher values protect smooth regions
                     from noise amplification.

    Returns:
        Detail-enhanced BGR uint8 frame.
    """
    # Step 1: Create the blurred version (the "unsharp" mask)
    blurred = cv2.GaussianBlur(frame, kernel_size, sigma)

    # Step 2: Apply the sharpening formula
    # cv2.addWeighted(src1, alpha, src2, beta, gamma)
    # → result = alpha * src1 + beta * src2 + gamma
    # → (1 + amount) * original + (-amount) * blurred
    sharpened = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)

    # Step 3: Apply threshold (skip low-contrast areas if threshold > 0)
    if threshold > 0:
        # Create mask: only apply sharpening where original and blurred differ significantly
        diff = np.abs(frame.astype(np.int32) - blurred.astype(np.int32))
        low_contrast_mask = diff.mean(axis=2) < threshold  # shape: (H, W)
        # Where low contrast: use original; where high contrast: use sharpened
        np.copyto(sharpened, frame, where=low_contrast_mask[:, :, np.newaxis])

    # Step 4: Clip to valid uint8 range
    return np.clip(sharpened, 0, 255).astype(np.uint8)
