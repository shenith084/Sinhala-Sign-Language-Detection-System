import json
import os

notebook_path = "c:/project/ssl400_research_project/SSL400_Master_Training.ipynb"

# The completely updated notebook structure
notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🚀 SSL400 Step-by-Step Kaggle Training\n",
    "Run this step-by-step manually so you can monitor all the logs and progress bars!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. INSTALL DEPENDENCIES\n",
    "!pip install tf-keras ultralytics --quiet\n",
    "print(\"✅ Dependencies Installed!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. UNZIP CODE & COPY SPLITS\n",
    "!cp -r /kaggle/input/ssl400-dataset/src /kaggle/working/\n",
    "!cp -r /kaggle/input/ssl400-dataset/data /kaggle/working/\n",
    "!cp /kaggle/input/ssl400-dataset/config.yaml /kaggle/working/\n",
    "print(\"✅ Code & Splits Copied!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. INJECT THE BUG FIX FOR EFFICIENTNETV2 NORMALIZATION\n",
    "file_path = '/kaggle/working/src/data/tf_dataset_builder.py'\n",
    "with open(file_path, 'r') as f:\n",
    "    code = f.read()\n",
    "\n",
    "fix = \"\"\"\n",
    "    # -----------------------------------------------------------------------\n",
    "    # CRITICAL BUG FIX: Un-normalize for EfficientNetV2\n",
    "    # -----------------------------------------------------------------------\n",
    "    def _unnormalize(frames, labels):\n",
    "        frames = (frames + 1.0) * 127.5\n",
    "        return frames, labels\n",
    "    \n",
    "    ds = ds.map(_unnormalize, num_parallel_calls=AUTOTUNE)\n",
    "\n",
    "    ds = ds.prefetch(AUTOTUNE)\n",
    "\n",
    "    return ds\n",
    "\"\"\"\n",
    "\n",
    "code = code.replace('    ds = ds.prefetch(AUTOTUNE)\\n\\n    return ds', fix)\n",
    "with open(file_path, 'w') as f:\n",
    "    f.write(code)\n",
    "print(\"✅ tf_dataset_builder.py Patched successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. SET YOUR EXPERIMENT ID MANUALLY\n",
    "# Change this from 1 to 4 to run each experiment.\n",
    "EXP_ID = 2\n",
    "print(f\"You are running Experiment {EXP_ID}!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5. PROCESS VIDEOS INTO ENHANCED FRAMES\n",
    "!python src/data/video_to_frames.py --exp_id {EXP_ID}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 6. TRAIN THE MODEL (PHASE 1 & PHASE 2)\n",
    "!python src/training/train.py --exp_id {EXP_ID}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 7. GENERATE CLASSIFICATION REPORT\n",
    "import os\n",
    "import tensorflow as tf\n",
    "import tf_keras as keras\n",
    "from src.models.efficientnet_builder import build_efficientnet_bilstm\n",
    "from src.data.tf_dataset_builder import build_dataset\n",
    "import yaml\n",
    "import pandas as pd\n",
    "from sklearn.metrics import classification_report\n",
    "import numpy as np\n",
    "\n",
    "with open(\"config.yaml\", \"r\") as f:\n",
    "    config = yaml.safe_load(f)\n",
    "\n",
    "test_ds = build_dataset(\n",
    "    split_csv=f\"data/splits/test.csv\",\n",
    "    processed_dir=f\"data/processed/exp{EXP_ID}\",\n",
    "    num_classes=8,\n",
    "    batch_size=2,\n",
    "    is_training=False,\n",
    "    use_augmentation=False\n",
    ")\n",
    "\n",
    "model_path = f\"models/experiment_{EXP_ID}/best_model_phase2.keras\"\n",
    "if os.path.exists(model_path):\n",
    "    empty_model = build_efficientnet_bilstm(8, [32, 224, 224, 3])\n",
    "    empty_model.load_weights(model_path)\n",
    "    \n",
    "    true_labels = []\n",
    "    predictions = []\n",
    "    for frames, labels in test_ds:\n",
    "        preds = empty_model.predict(frames, verbose=0)\n",
    "        true_labels.extend(np.argmax(labels.numpy(), axis=1))\n",
    "        predictions.extend(np.argmax(preds, axis=1))\n",
    "    \n",
    "    label_map = {0: \"Thank you\", 1: \"Hello\", 2: \"Good\", 3: \"House\", 4: \"Eat\", 5: \"Drink\", 6: \"Tell\", 7: \"Write\"}\n",
    "    target_names = [label_map[i] for i in range(8)]\n",
    "    \n",
    "    print(f\"\\n{'='*50}\")\n",
    "    print(f\"       EXP {EXP_ID} CLASSIFICATION REPORT\")\n",
    "    print(f\"{'='*50}\")\n",
    "    print(classification_report(true_labels, predictions, target_names=target_names, zero_division=0))\n",
    "else:\n",
    "    print(f\"❌ Model file not found at {model_path}.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 8. CLEANUP (RUN THIS BEFORE STARTING THE NEXT EXPERIMENT)\n",
    "# This deletes the processed frames so Kaggle doesn't run out of disk space!\n",
    "!rm -rf /kaggle/working/data/processed/*\n",
    "print(\"✅ Disk cleaned! You can now change EXP_ID at the top and run the next experiment.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
    
print("Notebook updated successfully!")
