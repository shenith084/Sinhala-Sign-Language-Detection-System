import sys
from pathlib import Path
import tensorflow as tf
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_MODEL = None
_TARGET_FRAMES = 32

def load_model(exp_id: int, config: dict):
    global _MODEL, _TARGET_FRAMES
    exp_config = config["experiments"][exp_id]
    model_path = Path(exp_config["model_dir"]) / "best_model.keras"
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Train or generate dummy model first.")
        sys.exit(1)
        
    logger.info(f"Loading Model for EXP{exp_id}: {exp_config['name']} ...")
    import tensorflow_hub as hub
    from src.models.i3d_builder import SSL400I3DModel
    _MODEL = tf.keras.models.load_model(str(model_path), custom_objects={
        'SSL400I3DModel': SSL400I3DModel,
        'KerasLayer': hub.KerasLayer
    }, compile=False)
    _TARGET_FRAMES = config["video"]["target_frames"]
    logger.info("Model loaded successfully.")

def get_model():
    return _MODEL

def get_target_frames() -> int:
    return _TARGET_FRAMES

def predict_batch(batch_clip: tf.Tensor):
    if _MODEL is None:
        raise ValueError("Model is not loaded.")
    return _MODEL.predict(batch_clip, verbose=0)[0]
