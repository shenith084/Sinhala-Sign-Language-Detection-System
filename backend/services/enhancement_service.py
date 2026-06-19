import sys
from pathlib import Path
from backend.utils.logger import get_logger

# Import enhancement factory from src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.enhancement.enhancement_factory import get_enhancer

logger = get_logger(__name__)

_ENHANCER_FN = None

def init_enhancer(exp_id: int, config_path: str = "config.yaml"):
    global _ENHANCER_FN
    try:
        _ENHANCER_FN = get_enhancer(exp_id, config_path)
        logger.info(f"Enhancement service initialized for EXP{exp_id}.")
    except Exception as e:
        logger.error(f"Failed to initialize enhancement service: {e}")
        sys.exit(1)

def apply_enhancement(video_clip):
    if _ENHANCER_FN is None:
        raise ValueError("Enhancer function not loaded.")
    
    import numpy as np
    enhanced_frames = []
    for frame in video_clip:
        enhanced_frames.append(_ENHANCER_FN(frame))
    return np.array(enhanced_frames)
