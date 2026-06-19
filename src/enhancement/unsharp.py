"""
unsharp.py
==========
EXP4 — Unsharp Masking (Detail Enhancement / Sharpening)

Addresses: Loss of fine-grained spatial features — finger separation,
           joint angles, palm texture — that differentiate similar signs.

Pipeline:
    1. Create a blurred "unsharp mask" via Gaussian Blur
    2. Compute: sharpened = original + amount × (original − blurred)
    3. Optional: apply threshold to protect flat regions from noise
    4. Clip to [0, 255] and cast back to uint8
    5. Resize + Normalize for I3D input

Academic Justification:
    Unsharp Masking is widely used in medical imaging (MRI, CT scans) to
    improve feature detectability in fine structures. Applying it to sign
    language gesture recognition is a valid novel contribution because:
    - Many Sinhala sign pairs differ only in subtle finger positions.
    - Standard convolutional filters may not detect these sub-pixel features.
    - Amplifying high-frequency edges before I3D feature extraction forces
      the model's Conv3D filters to encode more discriminative patterns.
    - The threshold parameter prevents noise amplification in background regions.
"""

import numpy as np
import cv2


def enhance_unsharp_masking(
    frame: np.ndarray,
    kernel_size: tuple = (5, 5),
    sigma: float = 1.0,
    amount: float = 1.5,
    threshold: int = 0,
    target_size: tuple = (224, 224),
) -> np.ndarray:
    """
    Apply Unsharp Masking to enhance fine spatial details in sign gestures.

    Formula: sharpened = original + amount * (original - blurred)
    Which equals: sharpened = (1 + amount) * original - amount * blurred

    Args:
        frame:       BGR frame as numpy uint8 array (H, W, 3)
        kernel_size: Gaussian blur kernel size (width, height).
                     (5,5) = moderate blur, captures mid-frequency details.
        sigma:       Gaussian blur sigma. 1.0 = standard deviation of 1px.
        amount:      Sharpening strength multiplier.
                     1.5 = aggressive; use 0.8–1.0 for subtle sharpening.
        threshold:   Minimum pixel difference to apply sharpening.
                     0 = sharpen everywhere.
                     10 = only sharpen edges > 10 intensity units apart.
        target_size: Output spatial size (width, height). Default (224, 224)

    Returns:
        Detail-enhanced, normalized float32 frame in range [-1, 1], shape (224, 224, 3) RGB

    Raises:
        ValueError: If frame is None or empty
    """
    if frame is None or frame.size == 0:
        raise ValueError("enhance_unsharp_masking received an empty or None frame.")

    # ── Step 1: Create blurred version (the "unsharp mask") ──────────────────
    blurred = cv2.GaussianBlur(frame, kernel_size, sigma)

    # ── Step 2: Compute sharpened image ──────────────────────────────────────
    # cv2.addWeighted: dst = alpha*src1 + beta*src2 + gamma
    # sharpened = (1 + amount)*original + (-amount)*blurred
    frame_int = frame.astype(np.float32)
    blurred_int = blurred.astype(np.float32)
    sharpened = cv2.addWeighted(
        frame_int, 1.0 + amount,
        blurred_int, -amount,
        0,
    )

    # ── Step 3: Apply threshold (optional noise protection) ──────────────────
    # In low-contrast regions (background), sharpening amplifies noise.
    # The threshold suppresses sharpening where edge intensity is below threshold.
    if threshold > 0:
        # Compute per-pixel mean absolute difference across channels
        diff = np.mean(np.abs(frame_int - blurred_int), axis=2)
        # Where diff < threshold, revert to original frame
        low_contrast_mask = diff < threshold
        sharpened[low_contrast_mask] = frame_int[low_contrast_mask]

    # ── Step 4: Clip and cast to uint8 ───────────────────────────────────────
    frame_sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # ── Step 5: Resize to I3D input size ─────────────────────────────────────
    frame_resized = cv2.resize(frame_sharpened, target_size, interpolation=cv2.INTER_LINEAR)

    # ── Step 6: Convert BGR → RGB + Normalize to [-1, 1] ─────────────────────
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_float = frame_rgb.astype(np.float32)
    frame_normalized = (frame_float / 127.5) - 1.0

    return frame_normalized
