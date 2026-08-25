import nbformat as nbf

nb = nbf.v4.new_notebook()

text0 = """# 🚀 SSL400 Step-by-Step Kaggle Training
Use this notebook to run your training step-by-step so you can see all the progress bars, logs, and plots!"""

code1 = """# 1. INSTALL DEPENDENCIES
!pip install tf-keras tf-models-official ultralytics --quiet
print("✅ Dependencies Installed!")"""

code2 = """# 2. UNZIP CODE
# IMPORTANT: Kaggle automatically unzips datasets, so we just copy the code over!
# Change 'ssl400-dataset' to whatever your dataset is named!
!cp -r /kaggle/input/ssl400-dataset/src /kaggle/working/
!cp -r /kaggle/input/ssl400-dataset/data /kaggle/working/
!cp /kaggle/input/ssl400-dataset/config.yaml /kaggle/working/
print("✅ Code & Splits Copied!")"""

code3 = """# 3. PROCESS RAW VIDEOS INTO ENHANCED FRAMES
# This will show a nice progress bar for all 785 videos!
EXP_ID = 4
!python src/data/video_to_frames.py --exp_id {EXP_ID}"""

code4 = """# 4. TRAIN THE MODEL (PHASE 1 & PHASE 2)
# You will see the Epoch-by-Epoch training output right here!
!python src/training/train.py --exp_id {EXP_ID}"""

code5 = """# 5. PLOT TRAINING CURVES
import pandas as pd
import matplotlib.pyplot as plt
import os

log1 = f"logs/experiment_{EXP_ID}/training_log_phase1.csv"
log2 = f"logs/experiment_{EXP_ID}/training_log_phase2.csv"

if os.path.exists(log1) and os.path.exists(log2):
    df1 = pd.read_csv(log1)
    df2 = pd.read_csv(log2)
    df2['epoch'] = df2['epoch'] + len(df1)
    df = pd.concat([df1, df2], ignore_index=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(df['epoch'], df['accuracy'], label='Train Acc', color='green', linewidth=2)
    axes[0].plot(df['epoch'], df['val_accuracy'], label='Val Acc', color='red', linestyle='--', linewidth=2)
    axes[0].axvline(x=len(df1)-1, color='blue', linestyle=':', label='Phase 2 Starts')
    axes[0].set_title(f'EXP {EXP_ID} Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend()
    
    axes[1].plot(df['epoch'], df['loss'], label='Train Loss', color='blue', linewidth=2)
    axes[1].plot(df['epoch'], df['val_loss'], label='Val Loss', color='orange', linestyle='--', linewidth=2)
    axes[1].axvline(x=len(df1)-1, color='blue', linestyle=':', label='Phase 2 Starts')
    axes[1].set_title(f'EXP {EXP_ID} Loss', fontsize=14, fontweight='bold')
    axes[1].legend()
    
    plt.show()
else:
    print("Logs not found yet!")"""

code6 = """# 6. DETAILED CLASSIFICATION REPORT
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tf_keras as keras

import sys
sys.path.insert(0, 'src')
from data.tf_dataset_builder import build_dataset

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

exp_config = [e for e in config["experiments"] if e["id"] == EXP_ID][0]
test_csv = "data/splits/test_split.csv"

print("Building Test Dataset...")
test_ds = build_dataset(
    split_csv=test_csv,
    processed_dir=exp_config["processed_dir"],
    num_classes=config["dataset"]["num_classes"],
    batch_size=2,
    is_training=False,
    num_frames=config["frames"]["num_frames"],
    target_size=(config["frames"]["width"], config["frames"]["height"])
)

model_path = f"models/experiment_{EXP_ID}/best_model_phase2.keras"
if os.path.exists(model_path):
    print(f"Loading Model: {model_path}...")
    model = keras.models.load_model(model_path)
    
    print("Generating Predictions...")
    preds = model.predict(test_ds)
    y_pred = np.argmax(preds, axis=1)
    
    test_df = pd.read_csv(test_csv)
    y_true = test_df['class_id'].values
    
    # Note: Use classes 0-7 from your actual map
    class_names = ["Thank you", "Hello", "Good", "House", "Eat", "Drink", "Tell", "Write"]
    
    print("\\n==================================================")
    print(f"       EXP {EXP_ID} CLASSIFICATION REPORT")
    print("==================================================")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
else:
    print("Model not found!")"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text0),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_code_cell(code5),
    nbf.v4.new_code_cell(code6)
]

with open('SSL400_Master_Training.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook generated!")
