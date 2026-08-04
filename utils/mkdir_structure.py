"""
mkdir_structure.py
==================
Creates the complete folder structure for the SSL400 Research Project.
Run this ONCE at the start of the project from the project root.

Usage:
    python mkdir_structure.py
"""

import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# All directories to create (relative to project root)
DIRECTORIES = [
    # Data
    "data/raw",
    "data/processed/exp1_baseline",
    "data/processed/exp2_clahe_gamma",
    "data/processed/exp3_bilateral",
    "data/processed/exp4_unsharp",
    "data/processed/exp5_hybrid",
    "data/splits",
    # Source
    "src/data",
    "src/enhancement",
    "src/models",
    "src/training",
    "src/evaluation",
    "src/visualization",
    "src/live_system",
    # Models
    "models/experiment_1",
    "models/experiment_2",
    "models/experiment_3",
    "models/experiment_4",
    "models/experiment_5",
    # Logs
    "logs/experiment_1",
    "logs/experiment_2",
    "logs/experiment_3",
    "logs/experiment_4",
    "logs/experiment_5",
    # Results
    "results/metrics",
    "results/figures",
    # Backend
    "backend/routes",
    "backend/services",
    "backend/utils",
    # Frontend
    "frontend/public",
    "frontend/src/pages",
    "frontend/src/components",
    "frontend/src/services",
    # Assets
    "assets/fonts",
]


def create_structure(base_dir: str = ".") -> None:
    """
    Creates the full project directory structure.

    Args:
        base_dir: Root directory of the project (default: current directory).
    """
    base = Path(base_dir).resolve()
    logger.info(f"Creating project structure under: {base}")

    created = 0
    existed = 0

    for rel_path in DIRECTORIES:
        full_path = base / rel_path
        if full_path.exists():
            logger.debug(f"[EXISTS]  {rel_path}")
            existed += 1
        else:
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"[CREATED] {rel_path}")
            created += 1

    # Create __init__.py files for Python packages
    python_packages = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/enhancement/__init__.py",
        "src/models/__init__.py",
        "src/training/__init__.py",
        "src/evaluation/__init__.py",
        "src/visualization/__init__.py",
        "src/live_system/__init__.py",
        "backend/__init__.py",
        "backend/routes/__init__.py",
        "backend/services/__init__.py",
        "backend/utils/__init__.py",
    ]

    for pkg in python_packages:
        pkg_path = base / pkg
        if not pkg_path.exists():
            pkg_path.touch()
            logger.info(f"[CREATED] {pkg}")
            created += 1
        else:
            existed += 1

    logger.info(f"\n✅ Structure complete: {created} created, {existed} already existed.")


if __name__ == "__main__":
    create_structure(".")
