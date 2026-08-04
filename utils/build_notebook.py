import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_code_cell("""\
# Cell 1 — Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
print('✅ Google Drive mounted at /content/drive')"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Cell 2 — Install Dependencies
!pip install tf-keras tf-models-official --quiet
!pip install opencv-python-headless Pillow tqdm pyyaml scikit-learn scipy --quiet
print('✅ Dependencies installed')"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Cell 3 — Setup Project
import os
import shutil

DRIVE_PROJECT_PATH = '/content/drive/MyDrive/SSL400_Research'
LOCAL_PROJECT_PATH = '/content/ssl400'

if os.path.exists(DRIVE_PROJECT_PATH):
    if os.path.exists(LOCAL_PROJECT_PATH):
        shutil.rmtree(LOCAL_PROJECT_PATH)
    shutil.copytree(DRIVE_PROJECT_PATH, LOCAL_PROJECT_PATH)
    print('✅ Project copied from Drive')
else:
    print('⚠️ Drive project not found.')

os.chdir(LOCAL_PROJECT_PATH)
print(f'📁 Working directory: {os.getcwd()}')"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Cell 4 — EXPERIMENT CONFIGURATION
EXP_ID = 1              # Experiment 1
BATCH_SIZE = 8          # Use 4 if GPU runs out of memory

DRIVE_MODEL_DIR = f'/content/drive/MyDrive/SSL400_Research/models/experiment_{EXP_ID}'
print(f'🔬 Experiment {EXP_ID}')"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Cell 5 — FORCE START FROM SCRATCH (Deletes all old models)
!rm -rf /content/ssl400/models/experiment_{EXP_ID}
!rm -rf /content/drive/MyDrive/SSL400_Research/models/experiment_{EXP_ID}
print('🗑️ Deleted old models. Training will start fresh from Phase 1, Epoch 1.')"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Cell 6 — GPU Check
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f'✅ GPU available: {[g.name for g in gpus]}')
else:
    print('⚠️ NO GPU detected!')"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Cell 7 — RUN TRAINING
!python /content/ssl400/src/training/train.py --exp_id={EXP_ID} --batch_size={BATCH_SIZE} --drive_dir={DRIVE_MODEL_DIR}"""))

with open('SSL400_Training_Colab.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
