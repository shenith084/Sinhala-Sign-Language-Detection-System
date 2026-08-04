import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

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
        self.model_fn = None
        self.active_exp_id = None
        self.model_lock = threading.Lock()
        self._initialized = True

    def load_model(self, exp_id: int, model_path: str) -> bool:
        """Load a SavedModel or .keras weights into memory in a thread-safe way."""
        path = Path(model_path)
        keras_path = path.parent / "best_model_phase2.keras"

        if not path.exists() and not keras_path.exists():
            logger.error(f"Model path not found: {path} and {keras_path}")
            return False

        with self.model_lock:
            try:
                logger.info(f"Loading EXP{exp_id} model...")
                
                if keras_path.exists():
                    logger.info(f"Loading .keras weights from {keras_path}")
                    from models.movinet_builder import build_model
                    # Build custom model architecture
                    self.model = build_model(num_classes=5)
                    try:
                        self.model.load_weights(str(keras_path), by_name=True, skip_mismatch=True)
                        logger.info("Successfully loaded phase2 weights (with skip_mismatch=True).")
                    except Exception as e2:
                        logger.warning(f"Failed to load phase2: {e2}. Trying phase1...")
                        phase1_path = path.parent / "best_model_phase1.keras"
                        self.model.load_weights(str(phase1_path), by_name=True, skip_mismatch=True)
                        logger.info("Successfully loaded phase1 weights (with skip_mismatch=True).")
                    
                    # Define a fast serving wrapper
                    @tf.function
                    def serve(x):
                        return self.model(x, training=False)
                    
                    self.model_fn = serve
                else:
                    logger.info(f"Loading SavedModel from {path}")
                    self.model = tf.saved_model.load(str(path))
                    self.model_fn = self.model.signatures["serving_default"]

                self.active_exp_id = exp_id
                
                # Warmup inference
                dummy = np.random.randn(1, 32, 224, 224, 3).astype(np.float32)
                self.model_fn(tf.constant(dummy))
                
                logger.info(f"Successfully loaded and warmed up EXP{exp_id} model.")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        """Run inference on a preprocessed tensor."""
        if self.model_fn is None:
            raise ValueError("No model loaded.")
            
        with self.model_lock:
            # Run the frozen graph
            outputs = self.model_fn(tf.constant(tensor))
            
            # Extract the logits array from the output dictionary or tensor
            if isinstance(outputs, dict):
                logits = list(outputs.values())[0]
            else:
                logits = outputs
            # Apply softmax
            probs = tf.nn.softmax(logits[0]).numpy()
            return probs
