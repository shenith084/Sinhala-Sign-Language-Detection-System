import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from enhancement.enhancement_factory import get_enhancer
from data.video_to_frames import smart_crop_frame
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
                
        # Initialize YOLOv8 model for dynamic cropping
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")
            logger.info("YOLOv8 initialized for real-time dynamic cropping.")
        except ImportError:
            logger.warning("ultralytics not installed. YOLO cropping disabled in live inference.")
            self.yolo_model = None

    def get_enhancer(self, exp_id: int):
        if exp_id not in self.enhancers:
            raise ValueError(f"No enhancer found for experiment {exp_id}")
        return self.enhancers[exp_id]

    def preprocess_frames(self, frames: list, exp_id: int, num_frames: int = 32, target_size=(224, 224)) -> np.ndarray:
        """
        Takes a list of raw BGR numpy frames, applies YOLOv8 cropping, enhancement, 
        samples, resizes, and normalizes into a tensor for model input.
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
            
            # Apply smart crop if detector is provided and not baseline
            if self.yolo_model is not None and exp_id != 1:
                frame = smart_crop_frame(frame, self.yolo_model)
                
            # Resize
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Apply enhancement
            try:
                frame = enhance_fn(frame)
            except Exception as e:
                logger.debug(f"Enhancement failed: {e}")
                
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # EfficientNetV2 expects [0, 255] range!
            frame = frame.astype(np.float32)
            processed.append(frame)
            
        tensor = np.stack(processed, axis=0)
        return tensor[np.newaxis, ...]  # Shape: (1, 32, 224, 224, 3)
