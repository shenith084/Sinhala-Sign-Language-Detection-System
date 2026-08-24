"""
baseline.py — Experiment 1: No Enhancement (Control Group)
============================================================
The baseline performs ONLY the mandatory pre-processing steps:
resize and normalize. No image enhancement is applied.

This is the control group. All other experiments are compared against this.
DO NOT add any enhancement here — it would break the single-variable rule.
"""

import numpy as np
import cv2


def enhance_baseline(frame: np.ndarray) -> np.ndarray:
    """
    Apply no image enhancement — return the raw frame as-is.

    The baseline experiment establishes the unmodified performance of
    MoViNet-A2 on SSL400 without any image quality improvements.
    Resize and normalization are applied later in the pipeline (video_to_frames.py)
    and are NOT considered enhancements.

    Args:
        frame: BGR uint8 numpy array (raw OpenCV frame).

    Returns:
        The same frame unchanged (BGR uint8).
    """
    return frame
