"""
enhancement_factory.py
======================
Factory function to retrieve the correct enhancement function by experiment ID.

This is the single point of control for the experiment variable.
All training and evaluation scripts call get_enhancer(exp_id) and
receive a callable that accepts a raw BGR frame and returns a
normalized float32 RGB frame ready for I3D input.

Usage:
    from src.enhancement.enhancement_factory import get_enhancer

    enhance_fn = get_enhancer(exp_id=2)
    processed_frame = enhance_fn(raw_bgr_frame)

Design Principle:
    Each enhancement function has the same signature:
        fn(frame: np.ndarray) -> np.ndarray
        Input:  BGR uint8 (H, W, 3)
        Output: RGB float32 normalized to [-1, 1], shape (224, 224, 3)

    This allows all training code to be enhancement-agnostic.
    Only config.yaml's `active_experiment` changes between runs.
"""

import logging
from functools import partial
from typing import Callable

import numpy as np
import yaml
from pathlib import Path

from .baseline    import preprocess_baseline
from .clahe_gamma import enhance_clahe_gamma
from .bilateral   import enhance_bilateral
from .unsharp     import enhance_unsharp_masking
from .hybrid      import enhance_hybrid

logger = logging.getLogger(__name__)

# ── Experiment ID → Function mapping ─────────────────────────────────────────
_EXPERIMENT_REGISTRY = {
    1: "EXP1_BASELINE",
    2: "EXP2_CLAHE_GAMMA",
    3: "EXP3_BILATERAL",
    4: "EXP4_UNSHARP",
    5: "EXP5_HYBRID",
}


def get_enhancer(
    exp_id: int,
    config_path: str = "config.yaml",
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factory function: returns the enhancement callable for a given experiment ID.

    The returned function signature is always:
        enhanced_frame = fn(raw_bgr_frame: np.ndarray) -> np.ndarray
        Output: float32 RGB normalized to [-1, 1], shape (224, 224, 3)

    All hyperparameters are loaded from config.yaml so they stay consistent.

    Args:
        exp_id:      Experiment ID (1–5)
        config_path: Path to config.yaml

    Returns:
        Callable that takes a BGR uint8 frame and returns normalized float32 RGB

    Raises:
        ValueError: If exp_id is not in range 1–5
    """
    if exp_id not in _EXPERIMENT_REGISTRY:
        raise ValueError(
            f"Invalid exp_id: {exp_id}. Must be one of {list(_EXPERIMENT_REGISTRY.keys())}"
        )

    # Load config for hyperparameter values
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    enh_cfg = config.get("enhancement", {})
    video_cfg = config.get("video", {})
    target_size = (video_cfg.get("frame_width", 224), video_cfg.get("frame_height", 224))

    label = _EXPERIMENT_REGISTRY[exp_id]
    logger.info(f"Enhancement factory: loading {label} (Experiment {exp_id})")

    if exp_id == 1:
        # EXP1 — Baseline: resize + normalize only
        fn = partial(preprocess_baseline, target_size=target_size)

    elif exp_id == 2:
        # EXP2 — CLAHE + Gamma Correction
        clahe_cfg = enh_cfg.get("clahe", {})
        fn = partial(
            enhance_clahe_gamma,
            clip_limit=clahe_cfg.get("clip_limit", 2.0),
            tile_grid=tuple(clahe_cfg.get("tile_grid", [8, 8])),
            gamma=clahe_cfg.get("gamma", 1.2),
            target_size=target_size,
        )

    elif exp_id == 3:
        # EXP3 — Bilateral Filter
        bil_cfg = enh_cfg.get("bilateral", {})
        fn = partial(
            enhance_bilateral,
            d=bil_cfg.get("d", 9),
            sigma_color=bil_cfg.get("sigma_color", 75.0),
            sigma_space=bil_cfg.get("sigma_space", 75.0),
            target_size=target_size,
        )

    elif exp_id == 4:
        # EXP4 — Unsharp Masking
        unsh_cfg = enh_cfg.get("unsharp", {})
        kernel = tuple(unsh_cfg.get("kernel_size", [5, 5]))
        fn = partial(
            enhance_unsharp_masking,
            kernel_size=kernel,
            sigma=unsh_cfg.get("sigma", 1.0),
            amount=unsh_cfg.get("amount", 1.5),
            threshold=unsh_cfg.get("threshold", 0),
            target_size=target_size,
        )

    elif exp_id == 5:
        # EXP5 — Hybrid (Bilateral → CLAHE → Unsharp)
        bil_cfg  = enh_cfg.get("bilateral", {})
        clahe_cfg = enh_cfg.get("clahe", {})
        hyb_cfg  = enh_cfg.get("hybrid", {})
        fn = partial(
            enhance_hybrid,
            bilateral_d=hyb_cfg.get("bilateral_d", 9),
            bilateral_sigma_color=hyb_cfg.get("bilateral_sigma", 75.0),
            bilateral_sigma_space=hyb_cfg.get("bilateral_sigma", 75.0),
            clahe_clip_limit=hyb_cfg.get("clahe_clip", 2.0),
            clahe_tile_grid=tuple(hyb_cfg.get("clahe_tile", [8, 8])),
            gamma=hyb_cfg.get("gamma", 1.2),
            unsharp_amount=hyb_cfg.get("unsharp_amount", 1.0),
            target_size=target_size,
        )

    logger.info(f"  [OK] Enhancer ready: {label}")
    return fn


def list_experiments() -> None:
    """Print all available experiment IDs and their enhancement labels."""
    print("\n  Available Experiments:")
    print("  " + "-" * 45)
    for exp_id, label in _EXPERIMENT_REGISTRY.items():
        print(f"  [{exp_id}] {label}")
    print("  " + "-" * 45 + "\n")


if __name__ == "__main__":
    list_experiments()
