import os
import sys
import threading
from pathlib import Path
import numpy as np
import tensorflow as tf

# Import MoViNet so Keras registers the custom layers
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

from utils.logger import logger

# Add src to path to import models if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

class ModelService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.model = None
        self.active_exp_id = None
        self.model_lock = threading.Lock()
        self._initialized = True

    def load_model(self, exp_id: int, model_path: str) -> bool:
        """Load a Keras model into memory in a thread-safe way."""
        path = Path(model_path)
        if not path.exists():
            logger.error(f"Model path not found: {path}")
            return False

        with self.model_lock:
            try:
                logger.info(f"Loading EXP{exp_id} model from {path}...")
                os.environ["TF_USE_LEGACY_KERAS"] = "1"
                try:
                    import tf_keras as keras
                except ImportError:
                    keras = tf.keras
                    
                self.model = keras.models.load_model(str(path))
                self.active_exp_id = exp_id
                
                # Warmup inference
                dummy = np.random.randn(1, 32, 224, 224, 3).astype(np.float32)
                self.model.predict(dummy, verbose=0)
                
                logger.info(f"✅ Successfully loaded and warmed up EXP{exp_id} model.")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        """Run inference on a preprocessed tensor."""
        if self.model is None:
            raise ValueError("No model loaded.")
            
        with self.model_lock:
            logits = self.model.predict(tensor, verbose=0)
            probs = tf.nn.softmax(logits[0]).numpy()
            return probs
