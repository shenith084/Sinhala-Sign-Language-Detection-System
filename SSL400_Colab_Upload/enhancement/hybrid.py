"""
hybrid.py — Experiment 4: Hybrid Combined Enhancement Pipeline
==============================================================
Sequentially applies all three enhancement techniques to maximize visual quality:
    Step 1: Bilateral Filter  → Denoise first (remove noise before amplifying)
    Step 2: CLAHE + Gamma     → Fix contrast on a clean image
    Step 3: Unsharp Masking   → Sharpen final details last

WHY this specific ORDER matters:
  (1) Bilateral FIRST: Denoising before contrast enhancement prevents CLAHE from
      amplifying background noise artifacts. Noise in → amplified noise out.
  (2) CLAHE SECOND: Applied to a clean image, CLAHE can enhance genuine contrast
      variations (hand-background boundary, skin tone) without amplifying noise.
  (3) Unsharp LAST: Sharpening after denoising avoids re-introducing noise through
      high-frequency amplification. A lower `amount=1.0` (vs 1.5 in EXP4) is
      used here to compensate for the additive effect of three sequential techniques.

This experiment tests the hypothesis that combining all three techniques yields
maximum performance and reveals whether technique interactions create positive
synergies or diminishing returns.
"""

import numpy as np
from enhancement.bilateral import enhance_bilateral
from enhancement.clahe_gamma import enhance_clahe_gamma
from enhancement.unsharp import enhance_unsharp_masking


def enhance_hybrid(
    frame: np.ndarray,
    bilateral_d: int = 9,
    bilateral_sigma: float = 75.0,
    clahe_clip: float = 2.0,
    clahe_tile: tuple = (8, 8),
    gamma: float = 1.2,
    unsharp_amount: float = 1.0
) -> np.ndarray:
    """
    Full hybrid enhancement pipeline: Bilateral → CLAHE+Gamma → Unsharp.

    Each technique compensates for a different visual degradation mode in SSL400:
    - Bilateral: Background/sensor noise
    - CLAHE+Gamma: Inconsistent lighting and low contrast
    - Unsharp: Blurry finger edges and low spatial detail

    Args:
        frame:           BGR uint8 numpy array (OpenCV frame).
        bilateral_d:     Bilateral filter neighborhood diameter (9 = strong).
        bilateral_sigma: Bilateral sigmaColor and sigmaSpace (75 = moderate).
        clahe_clip:      CLAHE clip limit (2.0 = balanced contrast enhancement).
        clahe_tile:      CLAHE tile grid size ((8,8) for 224×224 frames).
        gamma:           Gamma correction value (1.2 = gentle brightening).
        unsharp_amount:  Sharpening strength (1.0 = moderate, less than EXP4's 1.5
                         to avoid over-sharpening after bilateral denoising).

    Returns:
        Fully enhanced BGR uint8 frame.
    """
    # Step 1: Edge-preserving denoising
    frame = enhance_bilateral(
        frame,
        d=bilateral_d,
        sigma_color=bilateral_sigma,
        sigma_space=bilateral_sigma
    )

    # Step 2: Contrast and illumination normalization
    frame = enhance_clahe_gamma(
        frame,
        clip_limit=clahe_clip,
        tile_grid=clahe_tile,
        gamma=gamma
    )

    # Step 3: Detail amplification (moderate to avoid noise re-introduction)
    frame = enhance_unsharp_masking(
        frame,
        amount=unsharp_amount
    )

    return frame
