"""
bilateral_unsharp.py
====================
Edge-Preserving Sharpening (New Experiment 3)

Combines Bilateral Filtering (noise reduction) with Unsharp Masking (edge sharpening).
Mathematical rationale: 
  - Sharpening a raw image sharpens the background noise.
  - By applying a Bilateral Filter first, we smooth the background while keeping edges intact.
  - Then, Unsharp Masking sharpens those clean edges, resulting in highly defined hand boundaries.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path

# Load config once
config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Extract parameters
bf_d = config["enhancement"]["bilateral_d"]
bf_sc = config["enhancement"]["bilateral_sigma_color"]
bf_ss = config["enhancement"]["bilateral_sigma_space"]

us_k = config["enhancement"]["unsharp_kernel_size"]
us_s = config["enhancement"]["unsharp_sigma"]
us_a = config["enhancement"]["unsharp_amount"]

def enhance_bilateral_unsharp(frame: np.ndarray) -> np.ndarray:
    """
    Applies Bilateral Filtering followed by Unsharp Masking.

    Args:
        frame: BGR uint8 numpy array.

    Returns:
        Enhanced BGR uint8 numpy array.
    """
    # 1. Bilateral Filter (Smooth background, keep edges)
    smoothed = cv2.bilateralFilter(frame, d=bf_d, sigmaColor=bf_sc, sigmaSpace=bf_ss)
    
    # 2. Unsharp Masking (Sharpen the clean edges)
    kernel_size = (us_k, us_k)
    blurred = cv2.GaussianBlur(smoothed, kernel_size, us_s)
    
    # sharp = original + amount * (original - blurred)
    sharp = cv2.addWeighted(smoothed, 1.0 + us_a, blurred, -us_a, 0)
    
    return sharp
