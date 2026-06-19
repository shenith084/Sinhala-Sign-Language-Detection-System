"""
evaluate.py
===========
Phase 4 — Model Evaluation Script

Evaluates a trained I3D model on the test split. 
Generates and saves the classification report (Precision, Recall, F1-Score)
and overall accuracy to the results directory.

Usage:
    python src/evaluation/evaluate.py --exp_id 1
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, accuracy_score

# ── Environment Setup ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONIOENCODING"]      = "utf-8"

from src.data.tf_dataset_builder import build_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_experiment(exp_id: int, config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    exp_config = config["experiments"][exp_id]
    model_dir = Path(exp_config["model_dir"])
    results_dir = Path("results") / f"experiment_{exp_id}"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "best_model.keras"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Please run training first.")
        sys.exit(1)

    num_classes   = config["model"]["num_classes"]
    target_frames = config["video"]["target_frames"]
    seed          = config["project"]["seed"]

    logger.info(f"=== Evaluating Experiment {exp_id}: {exp_config['name']} ===")
    
    # ── Load Model ────────────────────────────────────────────────────────────
    logger.info(f"Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(str(model_path))

    # ── Load Test Data ────────────────────────────────────────────────────────
    logger.info("Building test dataset pipeline...")
    # Batch size can be larger during inference
    batch_size = 8 
    test_ds = build_dataset(
        split_csv="data/splits/test_split.csv",
        exp_id=exp_id,
        batch_size=batch_size,
        num_classes=num_classes,
        target_frames=target_frames,
        augment=False,
        shuffle=False,  # VERY IMPORTANT: Keep order for sklearn metrics
        seed=seed,
        config_path=config_path,
    )

    # Load ground truth directly from CSV for comparison
    test_df = pd.read_csv("data/splits/test_split.csv")
    y_true = test_df["class_id"].values

    # ── Run Inference ─────────────────────────────────────────────────────────
    logger.info(f"Running inference on {len(test_df)} test videos...")
    predictions = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(predictions, axis=1)

    # Note: tf.data.Dataset might drop samples or have different ordering if not careful.
    # Because we set drop_remainder=False and shuffle=False, y_pred matches y_true exactly.
    if len(y_pred) != len(y_true):
        logger.warning(
            f"Prediction count ({len(y_pred)}) does not match test set size ({len(y_true)}). "
            "Truncating ground truth for metric calculation."
        )
        y_true = y_true[:len(y_pred)]

    # ── Calculate Metrics ─────────────────────────────────────────────────────
    logger.info("Calculating metrics...")
    acc = accuracy_score(y_true, y_pred)
    
    # We use zero_division=0 to prevent warnings on missing classes in the test set
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, zero_division=0)

    # Extract macro averages
    macro_precision = report_dict["macro avg"]["precision"]
    macro_recall    = report_dict["macro avg"]["recall"]
    macro_f1        = report_dict["macro avg"]["f1-score"]
    weighted_f1     = report_dict["weighted avg"]["f1-score"]

    logger.info(f"Test Accuracy    : {acc:.4f}")
    logger.info(f"Macro Precision  : {macro_precision:.4f}")
    logger.info(f"Macro Recall     : {macro_recall:.4f}")
    logger.info(f"Macro F1-Score   : {macro_f1:.4f}")
    logger.info(f"Weighted F1-Score: {weighted_f1:.4f}")

    # ── Save Results ──────────────────────────────────────────────────────────
    # Save text report
    report_path = results_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment {exp_id} - {exp_config['name']}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report_str)
    
    # Save structured metrics to CSV
    metrics_df = pd.DataFrame({
        "Experiment_ID": [exp_id],
        "Enhancement": [exp_config["name"]],
        "Accuracy": [acc],
        "Macro_Precision": [macro_precision],
        "Macro_Recall": [macro_recall],
        "Macro_F1": [macro_f1],
        "Weighted_F1": [weighted_f1],
    })
    
    metrics_csv = results_dir / "summary_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    
    # Save raw predictions for confusion matrix generation
    raw_preds_path = results_dir / "raw_predictions.npz"
    np.savez_compressed(raw_preds_path, y_true=y_true, y_pred=y_pred, probabilities=predictions)

    logger.info(f"Results saved to {results_dir}")
    logger.info("Run `src/evaluation/confusion_matrix.py` next to generate plots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained I3D model")
    parser.add_argument("--exp_id", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    evaluate_experiment(args.exp_id, args.config)
