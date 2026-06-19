"""
baseline.py
===========
EXP1 — Baseline Enhancement (Control Group)

NO image enhancement is applied. Only mandatory I3D pre-processing:
  1. Resize to 224×224
  2. Normalize to [-1, 1]

Academic Note:
    This is intentionally zero-enhancement. Experiment 1 is the control
    group against which all other experiments are measured. Resize and
    normalize are NOT enhancements — they are required I3D input format.
"""

import numpy as np
import cv2
from typing import Optional


def preprocess_baseline(
    frame: np.ndarray,
    target_size: tuple = (224, 224),
) -> np.ndarray:
    """
    Baseline pre-processing: resize and normalize only. No enhancement.

    This is the CONTROL GROUP for the experiment. Any image passed through
    this function is only resized and normalized to meet I3D input specs.

    Args:
        frame:       BGR frame as numpy uint8 array (H, W, 3)
        target_size: Output spatial size (width, height). Default (224, 224)

    Returns:
        Normalized float32 frame in range [-1, 1], shape (224, 224, 3) RGB
    """
    if frame is None or frame.size == 0:
        raise ValueError("preprocess_baseline received an empty or None frame.")

    # Step 1: Resize to I3D input resolution
    frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

    # Step 2: Convert BGR → RGB (TensorFlow models expect RGB)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Step 3: Normalize to [-1, 1]  →  (pixel / 127.5) - 1.0
    frame_float = frame_rgb.astype(np.float32)
    frame_normalized = (frame_float / 127.5) - 1.0

    return frame_normalized
