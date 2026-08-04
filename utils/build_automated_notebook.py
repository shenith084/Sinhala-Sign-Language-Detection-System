import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# 🚀 SSL400 Training Notebook - Fully Automated"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 1 — Mount Google Drive\n",
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n",
            "print('✅ Google Drive mounted at /content/drive')"
        ]
    },

    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 2 — COPY PROJECT TO LOCAL GPU DISK (For maximum speed!)\n",
            "import shutil\n",
            "import os\n",
            "DRIVE_PROJECT_PATH = '/content/drive/MyDrive/SSL400_Research'\n",
            "LOCAL_PROJECT_PATH = '/content/ssl400'\n",
            "\n",
            "print('Copying code and config from Google Drive to local Colab memory...')\n",
            "os.makedirs(LOCAL_PROJECT_PATH, exist_ok=True)\n",
            "if os.path.exists(os.path.join(DRIVE_PROJECT_PATH, 'config.yaml')):\n",
            "    shutil.copy2(os.path.join(DRIVE_PROJECT_PATH, 'config.yaml'), LOCAL_PROJECT_PATH)\n",
            "\n",
            "local_src = os.path.join(LOCAL_PROJECT_PATH, 'src')\n",
            "if os.path.exists(local_src):\n",
            "    shutil.rmtree(local_src)\n",
            "shutil.copytree(os.path.join(DRIVE_PROJECT_PATH, 'src'), local_src)\n",
            "\n",
            "local_models = os.path.join(LOCAL_PROJECT_PATH, 'models')\n",
            "if os.path.exists(local_models):\n",
            "    shutil.rmtree(local_models)\n",
            "if os.path.exists(os.path.join(DRIVE_PROJECT_PATH, 'models')):\n",
            "    shutil.copytree(os.path.join(DRIVE_PROJECT_PATH, 'models'), local_models)\n",
            "\n",
            "local_data = os.path.join(LOCAL_PROJECT_PATH, 'data')\n",
            "if not os.path.exists(local_data):\n",
            "    os.symlink(os.path.join(DRIVE_PROJECT_PATH, 'data'), local_data)\n",
            "\n",
            "print('✅ Code successfully copied to /content/ssl400!')\n",
            "os.chdir(LOCAL_PROJECT_PATH)\n",
            "print(f'📁 Working directory: {os.getcwd()}')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 3 — INSTALL DEPENDENCIES\n",
            "!pip install tf-keras tf-models-official --quiet\n",
            "print('✅ Dependencies installed')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 4 — EXPERIMENT CONFIGURATION\n",
            "EXP_ID = 1              # Experiment 1 (Baseline)\n",
            "BATCH_SIZE = 4          # Recommended: 4 to prevent Out of Memory on Tesla T4\n",
            "DRIVE_MODEL_DIR = f'/content/drive/MyDrive/SSL400_Colab_Upload/models/experiment_{EXP_ID}'\n",
            "\n",
            "# RESUME SUPPORT ENABLED\n",
            "print('✅ Ready to resume training from latest checkpoint!')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 5 — PRE-PROCESS RAW VIDEOS INTO .NPY FRAMES\n",
            "!pip install ultralytics --quiet\n",
            "!python /content/ssl400/src/data/video_to_frames.py --exp_id {EXP_ID}"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 6 — RUN TRAINING\n",
            "import os\n",
            "os.environ['TF_USE_LEGACY_KERAS'] = '1'\n",
            "!python /content/ssl400/src/training/train.py --exp_id={EXP_ID} --batch_size={BATCH_SIZE} --drive_dir={DRIVE_MODEL_DIR}"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 7 — EVALUATE MODEL ON TEST SET (Precision, Recall, F1)\n",
            "!python /content/ssl400/src/evaluation/evaluate.py --exp_id={EXP_ID}"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 8 — SHOW TRAINING CURVES\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import os\n",
            "log_file = f'/content/ssl400/logs/experiment_{EXP_ID}/training_log_phase2.csv'\n",
            "if os.path.exists(log_file):\n",
            "    df = pd.read_csv(log_file)\n",
            "    plt.style.use('seaborn-v0_8-whitegrid')\n",
            "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
            "    ax1.plot(df['accuracy'], label='Train Acc', linewidth=2, color='#2ca02c')\n",
            "    ax1.plot(df['val_accuracy'], label='Val Acc', linewidth=2, color='#d62728', linestyle='--')\n",
            "    ax1.set_title(f'Accuracy Curve - Experiment {EXP_ID}', fontsize=14, pad=15)\n",
            "    ax1.legend()\n",
            "    ax2.plot(df['loss'], label='Train Loss', linewidth=2, color='#1f77b4')\n",
            "    ax2.plot(df['val_loss'], label='Val Loss', linewidth=2, color='#ff7f0e', linestyle='--')\n",
            "    ax2.set_title(f'Loss Curve - Experiment {EXP_ID}', fontsize=14, pad=15)\n",
            "    ax2.legend()\n",
            "    plt.show()\n",
            "else:\n",
            "    print('Training has not finished yet, so there is no graph to draw!')\n"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("c:/project/ssl400_research_project/SSL400_Training_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Successfully generated fully automated notebook!")
