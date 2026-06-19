"""
hybrid.py
=========
EXP5 — Hybrid Combined Enhancement Pipeline

Applies all three techniques in a carefully ordered sequence:
    Step 1: Bilateral Filter   → Denoise BEFORE amplifying contrast
    Step 2: CLAHE + Gamma      → Fix lighting on a CLEAN image
    Step 3: Unsharp Masking    → Amplify details LAST to avoid noise re-introduction

Sequence Justification:
    The ORDER of operations is critical to avoid compounding artifacts:
    - If CLAHE is applied first (on noisy image) → it amplifies noise textures.
    - If Unsharp Masking is applied first → it sharpens noise before denoising.
    - Bilateral FIRST removes noise, then CLAHE enhances real signal structure,
      then Unsharp Masking amplifies the refined edges, not noise.
    - The unsharp_amount is reduced to 1.0 (vs 1.5 in EXP4) to account for
      the fact that CLAHE already enhanced contrast — aggressive sharpening
      on top of CLAHE would over-process the image.

Academic Hypothesis Being Tested:
    "Combining all best individual techniques yields the maximum performance
     improvement, OR reveals diminishing returns / negative synergies."
    This is a key research contribution — the result either supports or
    challenges the assumption that more processing is always better.
"""

import numpy as np
import cv2


def enhance_hybrid(
    frame: np.ndarray,
    # Bilateral params
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 75.0,
    bilateral_sigma_space: float = 75.0,
    # CLAHE params
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: tuple = (8, 8),
    gamma: float = 1.2,
    # Unsharp params
    unsharp_kernel: tuple = (5, 5),
    unsharp_sigma: float = 1.0,
    unsharp_amount: float = 1.0,   # Reduced vs EXP4 (1.5) to avoid over-processing
    unsharp_threshold: int = 0,
    # Output
    target_size: tuple = (224, 224),
) -> np.ndarray:
    """
    Full hybrid enhancement pipeline:
        Step 1: Bilateral Filter  → Edge-preserving denoising
        Step 2: CLAHE + Gamma     → Adaptive contrast + brightness normalization
        Step 3: Unsharp Masking   → Fine spatial detail amplification

    Args:
        frame:                BGR frame as numpy uint8 array (H, W, 3)
        bilateral_d:          Bilateral filter neighborhood diameter
        bilateral_sigma_color: Bilateral color sigma
        bilateral_sigma_space: Bilateral spatial sigma
        clahe_clip_limit:     CLAHE contrast limit (2.0 = moderate)
        clahe_tile_grid:      CLAHE tile grid size
        gamma:                Gamma correction value (1.2 = gentle brightening)
        unsharp_kernel:       Gaussian kernel for unsharp mask creation
        unsharp_sigma:        Gaussian sigma for unsharp mask
        unsharp_amount:       Sharpening strength (1.0, reduced from EXP4's 1.5)
        unsharp_threshold:    Minimum edge intensity to sharpen (0 = everywhere)
        target_size:          Output spatial size (width, height). Default (224, 224)

    Returns:
        Fully enhanced, normalized float32 frame in range [-1, 1], shape (224, 224, 3) RGB

    Raises:
        ValueError: If frame is None or empty
    """
    if frame is None or frame.size == 0:
        raise ValueError("enhance_hybrid received an empty or None frame.")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Bilateral Filter — denoise while preserving edges
    # ─────────────────────────────────────────────────────────────────────────
    frame = cv2.bilateralFilter(
        frame,
        d=bilateral_d,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: CLAHE + Gamma — contrast and illumination normalization
    # ─────────────────────────────────────────────────────────────────────────
    # Convert to LAB, apply CLAHE to L channel only
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Apply gamma correction via LUT
    inv_gamma = 1.0 / gamma
    gamma_table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in range(256)],
        dtype=np.uint8,
    )
    frame = cv2.LUT(frame, gamma_table)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Unsharp Masking — amplify fine spatial details
    # ─────────────────────────────────────────────────────────────────────────
    blurred = cv2.GaussianBlur(frame, unsharp_kernel, unsharp_sigma)
    frame_int   = frame.astype(np.float32)
    blurred_int = blurred.astype(np.float32)
    sharpened   = cv2.addWeighted(
        frame_int, 1.0 + unsharp_amount,
        blurred_int, -unsharp_amount,
        0,
    )

    if unsharp_threshold > 0:
        diff = np.mean(np.abs(frame_int - blurred_int), axis=2)
        low_contrast_mask = diff < unsharp_threshold
        sharpened[low_contrast_mask] = frame_int[low_contrast_mask]

    frame = np.clip(sharpened, 0, 255).astype(np.uint8)

    # ─────────────────────────────────────────────────────────────────────────
    # Resize + BGR → RGB + Normalize to [-1, 1]
    # ─────────────────────────────────────────────────────────────────────────
    frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    frame_rgb     = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_float   = frame_rgb.astype(np.float32)
    frame_normalized = (frame_float / 127.5) - 1.0

    return frame_normalized
