"""
evaluate.py
===========
Load a trained model and compute all evaluation metrics on the test set.

Computes:
  - Top-1 Accuracy
  - Top-5 Accuracy
  - Macro Precision, Recall, F1-Score
  - Per-class F1 scores (for statistical significance analysis)
  - Training time (from CSV logs)
  - Inference latency (ms per sample)

Output:
  - results/metrics/experiment_{N}_metrics.json
  - results/metrics/all_experiments_metrics.csv (aggregated)

Usage:
    python src/evaluation/evaluate.py --exp_id 1
    python src/evaluation/evaluate.py --all   # evaluate all 5 experiments
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
try:
    import tf_keras as keras
except ImportError:
    keras = tf.keras

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_test_data(
    test_csv: str,
    processed_dir: str,
    num_frames: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all test set .npy files into memory.

    Args:
        test_csv:      Path to test_split.csv.
        processed_dir: Directory containing class_id/{stem}.npy files.
        num_frames:    Expected frames per clip.

    Returns:
        Tuple of (X: shape (N, 32, 224, 224, 3), y: shape (N,) integer labels).
    """
    df = pd.read_csv(test_csv)
    processed = Path(processed_dir)

    X_list, y_list = [], []
    missing = 0

    for _, row in df.iterrows():
        video_path = Path(row["video_path"])
        class_id = int(row["class_id"])
        npy_file = processed / str(class_id) / f"{video_path.stem}.npy"

        if npy_file.exists():
            frames = np.load(str(npy_file)).astype(np.float32)
            X_list.append(frames)
            y_list.append(class_id)
        else:
            missing += 1

    if missing > 0:
        logger.warning(f"{missing} .npy files missing from test set.")

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int32)


def compute_top5_accuracy(
    y_true: np.ndarray,
    logits: np.ndarray
) -> float:
    """
    Compute Top-5 accuracy: fraction of samples where the true label is among
    the top 5 predicted classes.

    Args:
        y_true:  Integer class labels, shape (N,).
        logits:  Raw model output, shape (N, num_classes).

    Returns:
        Top-5 accuracy as a float in [0, 1].
    """
    top5_preds = np.argsort(logits, axis=1)[:, -5:]
    correct = sum(
        y_true[i] in top5_preds[i]
        for i in range(len(y_true))
    )
    return correct / len(y_true)


def measure_inference_latency(
    model: keras.Model,
    num_frames: int = 32,
    num_warmup: int = 5,
    num_measure: int = 20
) -> float:
    """
    Measure average inference latency in milliseconds per single sample.

    Args:
        model:       Trained Keras model.
        num_frames:  Frames per sample.
        num_warmup:  Warmup runs (not measured).
        num_measure: Measured runs.

    Returns:
        Average inference time in milliseconds.
    """
    dummy = np.random.randn(1, num_frames, 224, 224, 3).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        model.predict(dummy, verbose=0)

    # Measure
    times = []
    for _ in range(num_measure):
        start = time.time()
        model.predict(dummy, verbose=0)
        times.append((time.time() - start) * 1000.0)

    return float(np.mean(times))


def get_training_time_minutes(log_dir: str) -> float:
    """
    Estimate total training time from CSV logs.

    Args:
        log_dir: Path to logs/experiment_{N}/ directory.

    Returns:
        Total training time in minutes (or -1 if not available).
    """
    log_path = Path(log_dir)
    total_time = 0.0
    found = False

    for phase in ["phase1", "phase2"]:
        csv_file = log_path / f"training_log_{phase}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            if "epoch_time" in df.columns:
                total_time += df["epoch_time"].sum() / 60.0
                found = True

    return total_time if found else -1.0


def evaluate_experiment(exp_id: int, config: dict) -> Dict:
    """
    Run full evaluation for one experiment.

    Args:
        exp_id:  Experiment ID (1–5).
        config:  Loaded config.yaml.

    Returns:
        Dict with all computed metrics.
    """
    exp = [e for e in config["experiments"] if e["id"] == exp_id][0]
    logger.info(f"\n{'='*60}")
    logger.info(f"  Evaluating EXP{exp_id}: {exp['name']}")
    logger.info(f"{'='*60}")

    model_path = PROJECT_ROOT / exp["model_dir"] / "best_model_phase2.keras"
    if not model_path.exists():
        logger.warning(f"No Phase 2 model found at {model_path}. Trying Phase 1...")
        model_path = PROJECT_ROOT / exp["model_dir"] / "best_model_phase1.keras"
        if not model_path.exists():
            logger.error(f"No model checkpoint found for EXP{exp_id}!")
            return {}

    # Load model properly by building architecture first
    logger.info("Building model architecture...")
    from src.models.movinet_builder import build_model
    model = build_model(
        num_classes=config["dataset"]["num_classes"],
        num_frames=config["frames"]["num_frames"],
        img_height=config["frames"]["height"],
        img_width=config["frames"]["width"],
        lstm_units=config["model"]["lstm_units"],
        dropout_rate=config["model"]["dropout_rate"]
    )
    logger.info(f"Loading weights from {model_path}...")
    model.load_weights(str(model_path))

    # Load test data on the fly from MP4s
    # CRITICAL: Read directly from Google Drive so we don't need to copy 10GB of videos!
    drive_root = Path("/content/drive/MyDrive/SSL400_Research")
    if drive_root.exists():
        test_csv = str(drive_root / config["paths"]["splits"] / "test_split.csv")
        raw_dir = str(drive_root / config["paths"]["raw_dataset"])
    else:
        test_csv = str(PROJECT_ROOT / config["paths"]["splits"] / "test_split.csv")
        raw_dir = str(PROJECT_ROOT / config["paths"]["raw_dataset"])
        
    num_frames = config["frames"]["num_frames"]
    batch_size = config["phase2"]["batch_size"]

    logger.info("Loading test data from MP4s...")
    from src.data.tf_dataset_builder import build_dataset
    from src.enhancement.enhancement_factory import get_enhancer

    test_ds = build_dataset(
        split_csv=test_csv,
        raw_dir=raw_dir,
        num_classes=config["dataset"]["num_classes"],
        batch_size=batch_size,
        is_training=False,
        enhance_fn=get_enhancer(exp_id),
        num_frames=num_frames,
        target_size=(config["frames"]["width"], config["frames"]["height"])
    )

    # Run inference
    logger.info("Running inference on test set...")
    logits = model.predict(test_ds, verbose=1)
    
    # Extract true labels
    y_true_list = []
    for _, y in test_ds: y_true_list.append(y.numpy())
    y_true = np.concatenate(y_true_list, axis=0)
    
    # Convert one-hot to integer labels if necessary
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)

    y_pred = np.argmax(logits, axis=1)

    # Metrics
    num_classes = config["dataset"]["num_classes"]

    top1_acc = accuracy_score(y_true, y_pred)
    top5_acc = compute_top5_accuracy(y_true, logits)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Latency
    latency_ms = measure_inference_latency(model, num_frames)

    # Training time from logs
    train_time = get_training_time_minutes(
        str(PROJECT_ROOT / exp["log_dir"])
    )

    results = {
        "exp_id": exp_id,
        "exp_name": exp["name"],
        "exp_label": exp["label"],
        "top1_accuracy": float(top1_acc),
        "top5_accuracy": float(top5_acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "inference_latency_ms": float(latency_ms),
        "training_time_min": float(train_time),
        "test_samples": int(len(y_test := y_true)),
        "per_class_f1": per_class_f1.tolist()
    }

    logger.info(f"\n  Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    logger.info(f"  Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    logger.info(f"  Macro F1:       {macro_f1:.4f}")
    logger.info(f"  Macro Precision:{macro_precision:.4f}")
    logger.info(f"  Macro Recall:   {macro_recall:.4f}")
    logger.info(f"  Latency:        {latency_ms:.1f} ms/sample")

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = PROJECT_ROOT / config["paths"]["results"] / "plots" / f"experiment_{exp_id}_cm.npy"
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cm_path, cm)
    
    # CRITICAL: Copy results back to Google Drive so they aren't lost!
    import shutil
    drive_root = Path("/content/drive/MyDrive/SSL400_Research")
    if drive_root.exists():
        drive_results = drive_root / "results"
        shutil.copytree(PROJECT_ROOT / "results", drive_results, dirs_exist_ok=True)
        logger.info(f"Results successfully backed up to Google Drive → {drive_results}")

    # Save per-experiment JSON
    metrics_dir = PROJECT_ROOT / config["paths"]["results"] / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / f"experiment_{exp_id}_metrics.json"
    with open(json_path, "w") as f:
        json.dump({k: v for k, v in results.items() if k != "per_class_f1"}, f, indent=2)
    logger.info(f"Saved metrics → {json_path}")

    np.save(str(metrics_dir / f"experiment_{exp_id}_confusion_matrix.npy"), cm)

    return results


def save_comparison_table(all_results: List[Dict], output_path: str) -> None:
    """
    Save the comparison table (thesis-ready format) as a CSV.

    Args:
        all_results:  List of metric dicts from evaluate_experiment().
        output_path:  Path to save the CSV.
    """
    rows = []
    for r in all_results:
        rows.append({
            "Experiment": f"EXP{r['exp_id']}",
            "Enhancement": r["exp_name"],
            "Top-1 Acc (%)": f"{r['top1_accuracy']*100:.2f}",
            "Top-5 Acc (%)": f"{r['top5_accuracy']*100:.2f}",
            "Macro F1": f"{r['macro_f1']:.4f}",
            "Precision": f"{r['macro_precision']:.4f}",
            "Recall": f"{r['macro_recall']:.4f}",
            "Train Time (min)": f"{r['training_time_min']:.1f}",
            "Inference (ms)": f"{r['inference_latency_ms']:.1f}"
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"\n📊 Comparison table saved → {output_path}")
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SSL400 experiment(s).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp_id", type=int, choices=[1, 2, 3, 4, 5],
                       help="Evaluate a single experiment")
    group.add_argument("--all", action="store_true",
                       help="Evaluate all 5 experiments")
    args = parser.parse_args()

    config = load_config()
    all_results = []

    if args.all:
        for eid in range(1, 6):
            result = evaluate_experiment(eid, config)
            if result:
                all_results.append(result)
    else:
        result = evaluate_experiment(args.exp_id, config)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        table_path = str(
            PROJECT_ROOT / config["paths"]["results"] / "metrics" /
            "all_experiments_metrics.csv"
        )
        save_comparison_table(all_results, table_path)


if __name__ == "__main__":
    main()
