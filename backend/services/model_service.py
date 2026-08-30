import os
import sys
import threading
from pathlib import Path
import numpy as np
import tensorflow as tf

# (MoViNet imports removed; using EfficientNetV2S)

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
        keras_path = path.parent / "fast_weights.weights.h5"

        if not path.exists() and not keras_path.exists():
            logger.error(f"Model path not found: {path} and {keras_path}")
            return False

        with self.model_lock:
            try:
                logger.info(f"Loading EXP{exp_id} model...")
                
                if keras_path.exists():
                    logger.info(f"Loading .keras weights from {keras_path}")
                    import yaml
                    import tensorflow as tf
                    
                    # Load config to get the correct parameters
                    with open(PROJECT_ROOT / "config.yaml", "r") as f:
                        config = yaml.safe_load(f)
                        
                    # Build custom model architecture matching training config EXACTLY,
                    # but WITHOUT custom names so it matches the Colab checkpoint exactly.
                    img_height = config["frames"]["height"]
                    img_width = config["frames"]["width"]
                    lstm_units = config["model"]["lstm_units"]
                    num_classes = config["dataset"]["num_classes"]
                    num_frames = config["frames"]["num_frames"]
                    
                    input_layer = tf.keras.Input(shape=(num_frames, img_height, img_width, 3))
                    base_model = tf.keras.applications.EfficientNetV2S(
                        input_shape=(img_height, img_width, 3),
                        include_top=False, weights=None, pooling='avg'
                    )
                    base_model._name = 'backbone'
                    
                    x = tf.keras.layers.TimeDistributed(base_model, name='efficientnetv2')(input_layer)
                    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
                    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units), name='bilstm')(x)
                    x = tf.keras.layers.Dropout(0.4)(x)
                    outputs = tf.keras.layers.Dense(num_classes, name='classifier_head')(x)
                    self.model = tf.keras.Model(inputs=input_layer, outputs=outputs)
                    
                    try:
                        self.model.load_weights(str(keras_path), skip_mismatch=True)
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
