# Sinhala Sign Language (SSL) Translation System 🇱🇰

An advanced, real-time Sinhala Sign Language translation system built on **I3D (Inflated 3D ConvNets)** with **Kinetics-400 Transfer Learning**. This repository implements a complete pipeline to recognize 383 unique Sinhala signs from video feeds using deep learning.

## 🌟 Key Features

- **383 Classes**: Extensive vocabulary of Sinhala words.
- **5 Video Enhancement Pipelines**: Built-in experimentation for Baseline, CLAHE, Bilateral Filtering, Unsharp Masking, and Hybrid enhancement.
- **Transfer Learning**: Pre-trained deepmind/i3d-kinetics-400 backbone frozen/fine-tuned.
- **Two-Phase Training**: Phase 1 (Warm-up) and Phase 2 (Fine-tuning).
- **Colab GPU Ready**: Fully packaged for Google Colab training.
- **Live Predictor**: OpenCV + PIL real-time webcam inference with Sinhala text rendering.
- **Flask REST API**: Backend API to serve the model to any web frontend.
- **Web App**: Premium glassmorphic frontend UI with Base64 frame extraction.
- **Sinhala TTS**: Auto-speaks recognized words using `gTTS`.

---

## 🛠️ Installation & Setup

We recommend using **Anaconda** to isolate dependencies.

### 1. Create Environment
```bash
conda create -n ssl400 python=3.10
conda activate ssl400
```

### 2. Install Dependencies
```bash
pip install tensorflow-cpu==2.16.1 tensorflow-hub==0.16.1 numpy==1.26.4
pip install opencv-python-headless pandas pyyaml scikit-learn matplotlib seaborn
pip install flask flask-cors gTTS pygame
```
> **Note on TensorFlow**: Do NOT install the `tensorflow` meta-package alongside `tensorflow-cpu` if it causes conflicts on Windows. This project strictly relies on `tensorflow-cpu` (or GPU equivalent in Colab).

---

## 🧠 Training the Model

Training is structured into 5 experiments. You can run a quick "sanity check" test on your laptop (CPU), but full training should be done on **Google Colab** (GPU).

### Laptop Quick Test (CPU)
Run a short 3-epoch warm-up test to verify the pipeline:
```bash
python src/training/train.py --exp_id 1 --batch_size 2
```

### Full Training (Google Colab GPU)
1. Upload this entire project folder to your Google Drive.
2. Open `SSL400_Training_Colab.ipynb` via Google Colab.
3. Select **Runtime -> Change runtime type -> T4 GPU**.
4. Run all cells. The notebook will automatically train all 5 experiments and save models to your Google Drive.

---

## 📊 Evaluation & Metrics

Once a model is trained (e.g., `best_model.keras` exists in `models/experiment_1/`), evaluate it on the test set:

1. **Calculate Accuracy & F1-Score**:
```bash
python src/evaluation/evaluate.py --exp_id 1
```
2. **Generate Confusion Matrix**:
```bash
python src/evaluation/confusion_matrix.py --exp_id 1 --top_k 30
```
3. **Cross-Experiment Comparison**:
```bash
python src/evaluation/statistical_analysis.py
```
*(Graphs will be saved to `results/figures/`)*

---

## 🎥 Running the Live Systems

You have two options for live webcam translation:

### Option A: Desktop OpenCV Predictor
A standalone desktop application that uses OpenCV to display the feed.
```bash
python src/live/live_predictor.py --exp_id 1
```

### Option B: Full Web Application (React + Flask)
A premium browser-based UI built with React, Vite, and TailwindCSS.

1. **Start the Flask Backend API:**
```bash
python backend/app.py --exp_id 1
```
2. **Start the React Frontend:**
```bash
cd frontend
npm run dev
```
3. Open `http://localhost:5173` in your browser. Allow camera permissions, click "Start Camera", and start signing!

---

## 📁 Project Structure

```text
ssl400_research_project/
├── backend/                  # Flask REST API for Web UI
│   └── app.py
├── config.yaml               # Master hyperparameters
├── data/                     # Data splits and word mappings
├── frontend/                 # Premium Glassmorphic Web App
│   ├── index.html
│   ├── script.js
│   └── style.css
├── logs/                     # TensorBoard & CSV logs
├── models/                   # Saved .keras weights
├── results/                  # Evaluation outputs and figures
├── src/                      # Core Codebase
│   ├── data/                 # tf.data pipeline
│   ├── enhancement/          # Video enhancement (CLAHE, Bilateral, etc.)
│   ├── evaluation/           # Accuracy, F1, Confusion Matrices
│   ├── live/                 # Desktop OpenCV predictor & TTS
│   ├── models/               # I3D Subclassed Model Builder
│   └── training/             # Two-Phase training script & callbacks
└── SSL400_Training_Colab.ipynb # Google Colab pipeline
```

## 📜 License
Research use only.
