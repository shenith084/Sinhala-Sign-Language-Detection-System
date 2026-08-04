import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# CELL 1: Setup & Mount Drive
setup_code = """# Cell 1 - SETUP & MOUNT DRIVE
import os
from google.colab import drive

print("Mounting Google Drive...")
drive.mount('/content/drive')

DRIVE_PROJECT_PATH = '/content/drive/MyDrive/SSL400_Research'
if not os.path.exists(DRIVE_PROJECT_PATH):
    print("ERROR: Could not find SSL400_Research in your Drive!")
    print("Please make sure you added the shortcut to your new account's Drive!")
else:
    print("✅ Successfully found SSL400_Research!")
"""
nb.cells.append(nbf.v4.new_code_cell(setup_code))

# CELL 2: Install dependencies
install_code = """# Cell 2 - INSTALL DEPENDENCIES
!pip install -q tf-models-official tf-keras
print("✅ Libraries installed!")
"""
nb.cells.append(nbf.v4.new_code_cell(install_code))

# CELL 3: Copy Code to Colab
copy_code = """# Cell 3 - COPY CODE TO LOCAL COLAB (FAST SPEED)
import shutil
import os

DRIVE_PROJECT_PATH = '/content/drive/MyDrive/SSL400_Research'
LOCAL_PROJECT_PATH = '/content/ssl400'

print("Copying code from Google Drive to local Colab memory...")
# We only copy src, config.yaml and models to save time
os.makedirs(LOCAL_PROJECT_PATH, exist_ok=True)

# Copy config
shutil.copy2(os.path.join(DRIVE_PROJECT_PATH, 'config.yaml'), LOCAL_PROJECT_PATH)

# Copy src
local_src = os.path.join(LOCAL_PROJECT_PATH, 'src')
if os.path.exists(local_src):
    shutil.rmtree(local_src)
shutil.copytree(os.path.join(DRIVE_PROJECT_PATH, 'src'), local_src)

# Copy models
local_models = os.path.join(LOCAL_PROJECT_PATH, 'models')
if os.path.exists(local_models):
    shutil.rmtree(local_models)
shutil.copytree(os.path.join(DRIVE_PROJECT_PATH, 'models'), local_models)

print("✅ Code successfully copied to /content/ssl400!")
"""
nb.cells.append(nbf.v4.new_code_cell(copy_code))

# CELL 4: Evaluate Model
eval_code = """# Cell 4 - EVALUATE EXPERIMENT 1 (Baseline)
# NOTE: Make sure your T4 GPU is enabled! (Runtime -> Change runtime type)

# Run the evaluation script
!python /content/ssl400/src/evaluation/evaluate.py --exp_id 1
"""
nb.cells.append(nbf.v4.new_code_cell(eval_code))

# Write notebook
notebook_path = "c:/project/ssl400_research_project/SSL400_Evaluate_Only.ipynb"
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Successfully generated evaluation notebook!")
