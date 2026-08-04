import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from enhancement.enhancement_factory import get_enhancer
from utils.logger import logger

class EnhancementService:
    def __init__(self):
        self.enhancers = {}
        # Pre-load all enhancer functions (1 to 4)
        for exp_id in range(1, 5):
            try:
                self.enhancers[exp_id] = get_enhancer(exp_id)
            except Exception as e:
                logger.warning(f"Failed to load enhancer for EXP{exp_id}: {e}")

    def get_enhancer(self, exp_id: int):
        if exp_id not in self.enhancers:
            raise ValueError(f"No enhancer found for experiment {exp_id}")
        return self.enhancers[exp_id]

    def preprocess_frames(self, frames: list, exp_id: int, num_frames: int = 32, target_size=(224, 224)) -> np.ndarray:
        """
        Takes a list of raw BGR numpy frames, applies enhancement, samples, 
        and normalizes into a tensor for model input.
        """
        enhance_fn = self.get_enhancer(exp_id)
        
        # Uniform sampling
        total = len(frames)
        if total == 0:
            raise ValueError("Empty frame list provided.")
            
        indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
        
        processed = []
        for idx in indices:
            frame = frames[idx].copy()
            # Resize FIRST for massive performance boost
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Apply enhancement
            try:
                frame = enhance_fn(frame)
            except Exception as e:
                logger.debug(f"Enhancement failed: {e}")
                
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Normalize [-1, 1]
            frame = frame.astype(np.float32)
            frame = (frame / 127.5) - 1.0
            processed.append(frame)
            
        tensor = np.stack(processed, axis=0)
        return tensor[np.newaxis, ...]  # Shape: (1, 32, 224, 224, 3)
