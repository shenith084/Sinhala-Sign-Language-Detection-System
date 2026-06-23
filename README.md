# SSL400 Sinhala Sign Language Research Project

This repository contains the complete implementation for the research project:
**"Performance Enhancement of Sinhala Sign Language Detection Systems Using Image Enhancement Techniques"**

## Project Architecture

The system uses a native **Keras 3 TimeDistributed MobileNetV3 + Bidirectional LSTM** architecture, evaluated across 5 experimental setups using different image enhancement pipelines to improve recognition accuracy on the low-resource SSL400 dataset.

## Setup Instructions

### 1. Local Environment Setup

1. Ensure you have Python 3.9+ installed.
2. Install dependencies (if you run locally):
   ```bash
   pip install -r requirements.txt
   ```
3. Generate the directory structure:
   ```bash
   python mkdir_structure.py
   ```

### 2. Training (Google Colab Recommended)

Due to hardware constraints, training should be performed on Google Colab using a T4 or A100 GPU.

1. Upload the `SSL400_Training_Colab.ipynb` notebook to Google Colab.
2. Mount your Google Drive.
3. Upload the `ssl400_research_project` folder to your Drive (e.g., to `/MyDrive/SSL400_Research`).
4. In the Colab notebook (Cell 4), set the `EXP_ID` to 1, 2, 3, 4, or 5.
5. Run all cells to begin training.
6. The training loop includes **resume support** and will save checkpoints back to your Google Drive automatically after every epoch.

### 3. Live Detection System

Once you have trained the models and downloaded the checkpoints to your local `models/` directory, you can run the live detection system.

**Start the Flask Backend API:**
```bash
cd backend
python app.py
```
*Note: The backend runs on `http://localhost:5000`*

**Start the React Frontend:**
```bash
cd frontend
npm install
npm run dev
```
*Note: The frontend runs on `http://localhost:5173`*

## The 5 Experiments

| Experiment ID | Enhancement Technique | Purpose |
| --- | --- | --- |
| **EXP 1** | Baseline | Control group; no enhancement applied. |
| **EXP 2** | CLAHE + Gamma | Improves local contrast and global illumination. |
| **EXP 3** | Bilateral Filter | Edge-preserving denoising. |
| **EXP 4** | Unsharp Masking | Amplifies fine spatial details and finger contours. |
| **EXP 5** | Hybrid | Sequentially applies Bilateral → CLAHE → Unsharp. |

## Documentation

The full task list and progress tracking is documented in the `task.md` file located in the `.gemini` brain directory, and the overarching plan is in `implementation_plan.md`.
