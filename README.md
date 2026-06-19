# Sinhala Sign Language (SSL400) Research System 🇱🇰

> **Research Title:** *Performance Enhancement of Sinhala Sign Language Detection Systems Using Image Enhancement Techniques*

An advanced, real-time Sinhala Sign Language translation system built on **I3D (Inflated 3D ConvNets)** with **Kinetics-400 Transfer Learning**. This repository implements a complete 5-experiment research pipeline to recognize 383 unique Sinhala signs from live video using deep learning.

**Target benchmark to beat:** 88.23% (Original SSL400 paper accuracy)

---

## 🌟 Key Features

| Feature | Detail |
|---|---|
| **383 Sign Classes** | Complete word-level Sinhala vocabulary |
| **5 Research Experiments** | Baseline, CLAHE+Gamma, Bilateral Filter, Unsharp Masking, Hybrid |
| **Two-Phase Training** | Phase 1 (Frozen backbone) + Phase 2 (Full fine-tuning) |
| **Mixup Augmentation** | Blends training samples to prevent overfitting |
| **Cosine Annealing LR** | Smooth LR decay for superior convergence |
| **Live Web UI** | Premium React + Flask real-time webcam inference |
| **Sinhala TTS** | Auto-speaks recognized words via gTTS |
| **Statistical Testing** | Paired t-test + Wilcoxon signed-rank validation |
| **Auto-Resume** | Training resumes automatically from last saved checkpoint |

---

## 🏗️ Architecture

```
Raw Video Frame
      │
      ▼
[Image Enhancement]   ← The experimental variable (changes per experiment)
      │
      ▼
[I3D Kinetics-400 Backbone]  ← Phase 1: FROZEN  |  Phase 2: UNFROZEN
      │
      ▼
[Dropout (0.5)] → [Dense(512, ReLU)] → [Dropout (0.3)] → [Dense(383, Softmax)]
      │
      ▼
Sinhala Sign Prediction (383 classes)
```

---

## 🔬 The 5 Research Experiments

| ID | Label | Enhancement Technique | Purpose |
|---|---|---|---|
| EXP1 | `BASELINE` | None (raw frames) | Control group — scientific reference |
| EXP2 | `CLAHE_GAMMA` | CLAHE + Gamma Correction | Fix indoor lighting & low contrast |
| EXP3 | `BILATERAL` | Bilateral Filter | Edge-preserving background noise reduction |
| EXP4 | `UNSHARP` | Unsharp Masking | Amplify finger detail & joint contours |
| EXP5 | `HYBRID` | Bilateral → CLAHE → Unsharp | Combined pipeline — expected best accuracy |

> **Critical rule:** Architecture, hyperparameters, data splits, and training procedure are **identical** across all 5 experiments. The **only** variable is the image enhancement technique.

---

## 🚀 Training Strategy (Upgraded — Two-Phase)

### Phase 1: Warm-Up (Frozen Backbone, 50 epochs)
- I3D backbone weights are **frozen** (Kinetics-400 weights preserved)
- Only the classification head is trained
- LR: `0.001` with **Cosine Annealing** schedule
- **Mixup Augmentation** (α=0.2) applied to training batches
- Expected accuracy: ~30–45%

### Phase 2: Fine-Tuning (Unfrozen Backbone, 30 epochs)
- I3D backbone is **unfrozen** — all weights trained
- Very low LR: `1e-5` with **Cosine Annealing** to prevent catastrophic forgetting
- Clean training data (no Mixup — precise gradients needed)
- Label smoothing reduced to `0.05`
- Expected accuracy jump: **+5% to +15% over Phase 1**
- **This is where we beat 88.23%**

---

## 🛠️ Local Setup

### 1. Create Anaconda Environment
```bash
conda create -n ssl400 python=3.10
conda activate ssl400
```

### 2. Install Dependencies
```bash
pip install tensorflow-cpu==2.16.1 tensorflow-hub==0.16.1 numpy==1.26.4
pip install opencv-python-headless pandas pyyaml scikit-learn matplotlib seaborn
pip install flask flask-cors gTTS
```

---

## 🎓 Training on Google Colab (GPU)

Full training requires a GPU and is done on **Google Colab** (free T4 GPU).

### Step-by-Step:
1. Upload this entire project folder to your **Google Drive** as `ssl400_research_project/`
2. Make sure your dataset videos are inside: `ssl400_research_project/SSL400/Dataset - Original/`
3. Open `SSL400_Training_Colab.ipynb` via Google Colab
4. Go to **Runtime → Change runtime type → T4 GPU**
5. In **Cell 2**, set `EXP_ID = 1` (change to 2, 3, 4, 5 for other experiments)
6. Click **▶ Run all**

### Auto-Resume Feature:
If your Colab session times out, **just run the notebook again**. It will automatically detect the saved checkpoint (`best_model_phase1.keras` or `best_model_phase2.keras`) and resume from exactly where it left off.

### Switching Between Experiments:
After EXP1 finishes and saves `best_model.keras`:
1. In Cell 2, change `EXP_ID = 2`
2. Click **Run all** — it starts a fresh Phase 1 brain for EXP2

---

## 📊 Generating Research Plots

After downloading the training logs from Google Drive to your local `logs/` folder:

```bash
# Generate plots for one experiment (Phase 1 + Phase 2 combined)
C:\Users\sheni\anaconda3\envs\ssl400\python.exe generate_report.py --exp_id 1

# Generate all-experiments comparison bar chart
C:\Users\sheni\anaconda3\envs\ssl400\python.exe generate_report.py --all
```

Output figures are saved to `results/figures/`:
- `exp1_two_phase_training.png` — Accuracy & Loss curves with SOTA benchmark line
- `all_experiments_comparison.png` — Bar chart comparing all 5 experiments

---

## 📡 Running the Live System

### Terminal 1 — Start Flask AI Backend:
```bash
# From project root
C:\Users\sheni\anaconda3\envs\ssl400\python.exe backend\app.py --exp_id 1
```
*(Change `--exp_id` to load a different experiment's model)*

### Terminal 2 — Start React Web UI:
```bash
cd frontend
npm run dev
```

### Open in browser:
**http://localhost:5173**

---

## 📈 Evaluation Commands

After training is complete and models are downloaded from Google Drive:

```bash
# Full evaluation metrics (Accuracy, Precision, Recall, F1)
C:\Users\sheni\anaconda3\envs\ssl400\python.exe src\evaluation\evaluate.py --exp_id 1

# Confusion matrix (top-30 classes)
C:\Users\sheni\anaconda3\envs\ssl400\python.exe src\evaluation\confusion_matrix.py --exp_id 1

# Statistical significance testing (compare all 5 experiments)
C:\Users\sheni\anaconda3\envs\ssl400\python.exe src\evaluation\statistical_analysis.py
```

---

## 📁 Project Structure

```
ssl400_research_project/
├── SSL400_Training_Colab.ipynb   ← Google Colab two-phase training notebook
├── generate_report.py            ← Publication-ready plot generator
├── config.yaml                   ← Master hyperparameters & paths
├── requirements.txt
│
├── backend/                      ← Flask REST API
│   ├── app.py
│   ├── routes/
│   │   ├── predict.py            ← POST /api/predict endpoint
│   │   ├── metrics.py            ← GET /api/metrics endpoint
│   │   └── experiments.py
│   ├── services/
│   │   ├── model_service.py      ← Keras model loader
│   │   └── enhancement_service.py
│   └── utils/
│       ├── sinhala_dictionary.py ← class_id → word mapping
│       └── logger.py
│
├── frontend/                     ← React + Vite Web App
│   └── src/
│       ├── pages/
│       │   ├── HomePage.jsx
│       │   ├── LiveDetectionPage.jsx
│       │   └── ExperimentsPage.jsx
│       ├── components/
│       │   ├── WebcamFeed.jsx
│       │   └── SinhalaTextDisplay.jsx
│       └── services/api.js
│
├── src/                          ← Core Python Research Code
│   ├── data/
│   │   ├── tf_dataset_builder.py ← tf.data video pipeline
│   │   ├── dataset_scanner.py
│   │   └── generate_splits.py
│   ├── enhancement/              ← Image enhancement functions
│   │   ├── baseline.py           ← EXP1: No enhancement
│   │   ├── clahe_gamma.py        ← EXP2: CLAHE + Gamma
│   │   ├── bilateral.py          ← EXP3: Bilateral Filter
│   │   ├── unsharp.py            ← EXP4: Unsharp Masking
│   │   ├── hybrid.py             ← EXP5: Bilateral+CLAHE+Unsharp
│   │   └── enhancement_factory.py
│   ├── models/
│   │   └── i3d_builder.py        ← SSL400I3DModel (subclassed Keras)
│   ├── training/
│   │   ├── train.py
│   │   └── callbacks.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── confusion_matrix.py
│   │   └── statistical_analysis.py
│   └── live/
│       ├── live_predictor.py     ← Desktop OpenCV predictor
│       └── text_to_speech.py
│
├── data/splits/                  ← Fixed train/val/test splits
│   ├── train_split.csv
│   ├── val_split.csv
│   ├── test_split.csv
│   └── sinhala_word_map.csv      ← class_id → English label
│
├── models/                       ← Trained .keras model weights
│   ├── experiment_1/best_model.keras
│   └── ...
│
└── logs/                         ← Training CSV logs per experiment
    ├── experiment_1/training_log_phase1.csv
    ├── experiment_1/training_log_phase2.csv
    └── experiment_1/final_test_results.txt
```

---

## 📜 Citation

If you use this work, please cite:
```
SSL400 Sinhala Sign Language Research System
Performance Enhancement of Sinhala Sign Language Detection Systems
Using Image Enhancement Techniques
Undergraduate Final-Year Research Project, 2024
```

## 📜 License
Research use only.
