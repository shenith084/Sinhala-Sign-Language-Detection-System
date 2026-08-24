"""
mobilenet_builder.py
====================
Builds the SSL400 sign language recognition model using a TimeDistributed MobileNetV3Large
followed by an LSTM layer.

Architecture:
    Input (batch, frames, 224, 224, 3)
    → TimeDistributed(MobileNetV3Large(include_top=False, weights='imagenet', pooling='avg'))
    → LSTM(lstm_units)
    → Dropout(dropout_rate)
    → Dense(num_classes)

Two-Phase Training:
    Phase 1: MobileNetV3 backbone FROZEN → train LSTM and head
    Phase 2: Full model UNFROZEN (except BatchNorm layers) → fine-tune everything
"""

import logging
import os
import math

# Use Legacy Keras 2 for consistency with the rest of the project
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

try:
    import tf_keras as keras
    logger_init = logging.getLogger(__name__)
    logger_init.info("Using tf_keras (Legacy Keras 2 API) in MobileNet builder.")
except ImportError:
    keras = tf.keras
    logger_init = logging.getLogger(__name__)
    logger_init.warning("tf_keras not found; falling back to tf.keras.")

logger = logging.getLogger(__name__)


def build_model(
    num_classes: int,
    num_frames: int = 32,
    img_height: int = 224,
    img_width: int = 224,
    lstm_units: int = 512,
    dropout_rate: float = 0.4
) -> keras.Model:
    """
    Build the SSL400 video classification model using MobileNetV3Large + LSTM.
    """
    input_layer = keras.Input(shape=(num_frames, img_height, img_width, 3), name="image")

    # 1. Base Model: MobileNetV3Large (ImageNet weights)
    base_model = keras.applications.MobileNetV3Large(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    # Give the backbone a specific name so we can find it easily
    base_model._name = 'backbone'

    # Phase 1 starts with a frozen backbone
    base_model.trainable = False

    # 2. Wrap in TimeDistributed
    x = keras.layers.TimeDistributed(base_model)(input_layer)

    # 3. Temporal Modeling with LSTM
    x = keras.layers.LSTM(lstm_units)(x)

    # 4. Classification Head
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, name="classifier_head")(x)

    model = keras.Model(inputs=input_layer, outputs=outputs, name="mobilenet_lstm_classifier")
    
    # Store reference to backbone to easily unfreeze later
    model.backbone = base_model
    
    return model


def compile_phase1(
    model: keras.Model,
    learning_rate: float = 1e-3,
    num_classes: int = 5,
    label_smoothing: float = 0.1
) -> keras.Model:
    """Compile Phase 1: Backbone FROZEN, train only LSTM and Head."""
    
    # Ensure backbone is frozen
    model.backbone.trainable = False

    logger.info("\n--- PHASE 1: Frozen backbone warm-up ---")
    logger.info("Phase 1: MobileNetV3 backbone FROZEN.")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True,  # Model outputs raw logits
            label_smoothing=label_smoothing
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy")
        ]
    )

    trainable = sum(1 for l in model.layers if l.trainable)
    logger.info(f"Phase 1 compile done. Trainable layers (Top-level): {trainable}")
    return model


def compile_phase2(
    model: keras.Model,
    learning_rate: float = 1e-4,
    num_classes: int = 5,
    label_smoothing: float = 0.1
) -> keras.Model:
    """Compile Phase 2: Full fine-tuning (except BatchNorm layers)."""
    
    logger.info("\n--- PHASE 2: Full fine-tuning ---")
    
    # Unfreeze the model
    model.trainable = True
    model.backbone.trainable = True
    
    # Freeze BatchNormalization layers to prevent validation accuracy collapse
    for layer in model.backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    logger.info("Phase 2: Unfroze MobileNetV3. BN layers remain FROZEN.")

    # Using AdamW for fine-tuning
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-4,
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
    schedule_type: str = "CosineDecay",
    initial_step: int = 0
) -> keras.optimizers.schedules.LearningRateSchedule:
    """Create a learning rate schedule."""
    if schedule_type == "CosineDecay":
        remaining_steps = max(1, decay_steps - initial_step)
        
        warmup_steps = int(decay_steps * 0.1)
        
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-6,
            decay_steps=remaining_steps,
            alpha=0.0,
            warmup_target=initial_lr,
            warmup_steps=warmup_steps
        )
    else:
        return initial_lr
