import tensorflow as tf
import os
from pathlib import Path
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

# Experiment 1
keras_path_1 = "models/experiment_1/best_model_phase2.keras"
save_path_1 = "models/experiment_1/saved_model"

if os.path.exists(keras_path_1) and not os.path.exists(save_path_1):
    print("Converting experiment_1 .keras to saved_model...")
    model = tf.keras.models.load_model(keras_path_1)
    model.export(save_path_1)
    print("Experiment 1 conversion successful!")

# Experiment 2
keras_path_2 = "models/experiment_2/best_model_phase2.keras"
save_path_2 = "models/experiment_2/saved_model"

if os.path.exists(keras_path_2) and not os.path.exists(save_path_2):
    print("Converting experiment_2 .keras to saved_model...")
    model = tf.keras.models.load_model(keras_path_2)
    model.export(save_path_2)
    print("Experiment 2 conversion successful!")
