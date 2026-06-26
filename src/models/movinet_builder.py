"""
movinet_builder.py
==================
Builds the SSL400 sign language recognition model using the OFFICIAL 
tf-models-official Model Garden implementation of MoViNet-A2.

CRITICAL BUG FIXES FROM MASTER PROMPT:
  1. Keras 3 Legacy Bug: Fixed by os.environ["TF_USE_LEGACY_KERAS"] = "1" 
     and importing tf_keras.
  2. Kinetics-600 Shape Mismatch: Fixed via CheckpointWrapper(tf.Module) 
     that isolates the backbone, ignoring the classifier head.
  3. from_logits Bug: Fixed in compile_phase1/2 by setting 
     CategoricalCrossentropy(from_logits=True).

Architecture:
    Input (batch, 32, 224, 224, 3)
    → MoViNet-A2 Backbone (Causal Conv3D)
    → Dropout(0.4)
    → Dense(150 classes)  ← Raw logits

Two-Phase Training:
    Phase 1: MoViNet backbone FROZEN → train head
    Phase 2: Full model UNFROZEN → fine-tune everything at LR=1e-5
"""

import logging
import os
from pathlib import Path
from typing import Tuple

import yaml

# ---------------------------------------------------------------------------
# FIX 1: The Keras 3 Legacy Bug (AttributeError: '_distribute_strategy')
# ---------------------------------------------------------------------------
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
try:
    import tf_keras as keras
    logger_init = logging.getLogger(__name__)
    logger_init.info("Using tf_keras (Legacy Keras 2 API).")
except ImportError:
    keras = tf.keras
    logger_init = logging.getLogger(__name__)
    logger_init.warning("tf_keras not found; falling back to tf.keras.")

try:
    from official.projects.movinet.modeling import movinet
    from official.projects.movinet.modeling import movinet_model
except ImportError:
    logger_init.warning("tf-models-official is not installed. MoViNet cannot be built.")

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# FIX 2: The Kinetics-600 Shape Mismatch Bug
# ---------------------------------------------------------------------------
class CheckpointWrapper(tf.Module):
    """
    Wrapper to isolate the backbone and force TF to ignore the classifier 
    head during Kinetics-600 weight restoration.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone


def load_kinetics_weights(model: keras.Model, checkpoint_dir: str):
    """Restore Kinetics-600 weights into the backbone only."""
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(f"CRITICAL: No Kinetics-600 checkpoint found in {checkpoint_dir}! Pretraining failed.")
        
    wrapper = CheckpointWrapper(model.backbone)
    checkpoint = tf.train.Checkpoint(model=wrapper)
    
    status = checkpoint.restore(latest)
    status.expect_partial()  # Ignore the 600-class head mismatch
    logger.info("Successfully loaded Kinetics-600 weights into backbone.")


def build_model(
    num_classes: int,
    num_frames: int = 32,
    img_height: int = 224,
    img_width: int = 224,
    lstm_units: int = 512,  # Not used in pure MoViNet, but kept for signature
    dropout_rate: float = 0.4
) -> keras.Model:
    """
    Build the SSL400 video classification model using MoViNet-A2.
    """
    # 1. Build the official MoViNet-A2 Backbone (Causal)
    backbone = movinet.Movinet(
        model_id='a2',
        causal=True,
        conv_type='3d',
        se_type='3d',
        activation='swish',
        gating_activation='sigmoid'
    )
    backbone.trainable = False  # Phase 1: frozen

    # 2. Build the official Classifier Model wrapper
    model = movinet_model.MovinetClassifier(
        backbone=backbone,
        num_classes=num_classes,
        output_states=False,
        dropout_rate=dropout_rate
    )

    # 3. Build the model shape
    model.build([None, num_frames, img_height, img_width, 3])
    logger.info(f"Official MoViNet-A2 built: {model.count_params():,} total parameters.")

    return model


def compile_phase1(
    model: keras.Model,
    learning_rate: float = 1e-3,
    num_classes: int = 150,
    label_smoothing: float = 0.1
) -> keras.Model:
    """
    Compile the model for Phase 1 (frozen backbone training).
    """
    # Ensure backbone is frozen
    model.backbone.trainable = False
    logger.info("Phase 1: MoViNet backbone FROZEN.")

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0 
    )

    # ---------------------------------------------------------------------------
    # FIX 3: The `from_logits` Mathematical Bug
    # ---------------------------------------------------------------------------
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True,  # CRITICAL: model outputs raw logits
            label_smoothing=label_smoothing
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy")
        ]
    )

    trainable = sum(1 for l in model.layers if l.trainable)
    logger.info(f"Phase 1 compile done. Trainable layers: {trainable}")
    return model


def compile_phase2(
    model: keras.Model,
    learning_rate: float = 1e-5,
    num_classes: int = 150,
    label_smoothing: float = 0.1
) -> keras.Model:
    # -------------------------------------------------------------------------
    # BULLETPROOF PARTIAL UNFREEZE (Block 4 + Head Only)
    # -------------------------------------------------------------------------
    # 1. Unfreeze the entire hierarchy first
    model.trainable = True
    if hasattr(model, 'backbone'):
        model.backbone.trainable = True
        
    # 2. Iterate through ALL nested submodules precisely
    for module in model.submodules:
        # Freeze lower-level 3D convolutions to prevent catastrophic gradient collapse
        if any(name in module.name for name in ["stem", "block0", "block1", "block2", "block3"]):
            module.trainable = False
            
    logger.info("Phase 2: Partial Unfreeze. ONLY block4 and head are trainable. BN layers are UNFROZEN.")

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=label_smoothing
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy")
        ]
    )

    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights
    )
    logger.info(f"Phase 2 compile done. "
                f"Trainable: {trainable_params:,} / {total_params:,} parameters.")
    return model


def get_lr_schedule(
    initial_lr: float,
    decay_steps: int,
    schedule_type: str = "CosineDecay"
) -> keras.optimizers.schedules.LearningRateSchedule:
    """Create a learning rate schedule."""
    if schedule_type == "CosineDecay":
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            alpha=0.0 
        )
    else:
        return initial_lr
