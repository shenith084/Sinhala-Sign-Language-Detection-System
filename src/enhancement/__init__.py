# src/enhancement/__init__.py
"""
Enhancement package for SSL400 research project.

Provides frame-level image enhancement functions for 5 experiments:
  - baseline   (EXP1): No enhancement
  - clahe_gamma(EXP2): CLAHE + Gamma Correction
  - bilateral  (EXP3): Bilateral Filter
  - unsharp    (EXP4): Unsharp Masking
  - hybrid     (EXP5): Bilateral → CLAHE → Unsharp (combined)
"""

from .baseline import preprocess_baseline
from .clahe_gamma import enhance_clahe_gamma
from .bilateral import enhance_bilateral
from .unsharp import enhance_unsharp_masking
from .hybrid import enhance_hybrid
from .enhancement_factory import get_enhancer

__all__ = [
    "preprocess_baseline",
    "enhance_clahe_gamma",
    "enhance_bilateral",
    "enhance_unsharp_masking",
    "enhance_hybrid",
    "get_enhancer",
]
