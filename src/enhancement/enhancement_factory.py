"""
enhancement_factory.py
======================
Factory function that returns the correct enhancement callable for each experiment.

Usage:
    from enhancement.enhancement_factory import get_enhancer

    enhance_fn = get_enhancer(exp_id=2)
    enhanced_frame = enhance_fn(raw_bgr_frame)
"""

from typing import Callable
import numpy as np


def get_enhancer(exp_id: int) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return the frame enhancement function for the given experiment ID.

    Each experiment uses a different image enhancement pipeline applied
    identically to every frame before it is fed into the model.

    Args:
        exp_id: Integer experiment identifier (1–5).
            1 = Baseline     (no enhancement)
            2 = CLAHE + Gamma Correction
            3 = Bilateral Filter
            4 = Unsharp Masking
            5 = Hybrid (Bilateral → CLAHE → Unsharp)

    Returns:
        A callable with signature: (frame: np.ndarray) -> np.ndarray
        where frame is a BGR uint8 numpy array.

    Raises:
        ValueError: If exp_id is not in the range 1–5.
    """
    if exp_id == 1:
        from enhancement.baseline import enhance_baseline
        return enhance_baseline

    elif exp_id == 2:
        from enhancement.clahe_gamma import enhance_clahe_gamma
        return enhance_clahe_gamma

    elif exp_id == 3:
        from enhancement.bilateral import enhance_bilateral
        return enhance_bilateral

    elif exp_id == 4:
        from enhancement.unsharp import enhance_unsharp_masking
        return enhance_unsharp_masking

    elif exp_id == 5:
        from enhancement.hybrid import enhance_hybrid
        return enhance_hybrid

    else:
        raise ValueError(
            f"Invalid exp_id: {exp_id}. Must be an integer between 1 and 5."
        )
