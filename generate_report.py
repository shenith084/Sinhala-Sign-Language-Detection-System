"""
generate_report.py
==================
Generates publication-ready research plots from Colab training logs.

Supports Two-Phase training logs (Phase 1 Warm-Up + Phase 2 Fine-Tuning)
with Mixup Augmentation and Cosine Annealing learning rate schedule.

Usage:
    python generate_report.py               # EXP1 by default
    python generate_report.py --exp_id 2   # For Experiment 2

Download the following files from Google Drive before running:
    logs/experiment_N/training_log_phase1.csv
    logs/experiment_N/training_log_phase2.csv   (after Phase 2 finishes)
    logs/experiment_N/final_test_results.txt
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Experiment metadata for axis labels
EXP_NAMES = {
    1: "Baseline (No Enhancement)",
    2: "CLAHE + Gamma Correction",
    3: "Bilateral Filter",
    4: "Unsharp Masking",
    5: "Hybrid (Bilateral + CLAHE + Unsharp)",
}

SOTA_ACCURACY = 0.8823   # SSL400 original paper benchmark to beat


def load_phase_log(exp_id: int, phase: int) -> pd.DataFrame | None:
    """Load a training log CSV for a given experiment and phase."""
    csv_path = Path(f"logs/experiment_{exp_id}/training_log_phase{phase}.csv")
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["epoch"] = df["epoch"] + 1  # Keras logs 0-indexed epochs
    return df


def print_test_results(exp_id: int):
    """Print final held-out test set results if available."""
    results_path = Path(f"logs/experiment_{exp_id}/final_test_results.txt")
    if results_path.exists():
        print("\n" + "=" * 60)
        print(f"  FINAL TEST RESULTS — EXP{exp_id}")
        print("=" * 60)
        print(results_path.read_text())
        print("=" * 60)
    else:
        print(f"\n[INFO] No final_test_results.txt found for EXP{exp_id}.")
        print(f"       Download from: Google Drive -> logs/experiment_{exp_id}/")


def generate_research_plots(exp_id: int = 1) -> None:
    """
    Generate publication-quality training curves for a given experiment.

    Combines Phase 1 (frozen backbone) and Phase 2 (fine-tuned backbone) logs
    into a single dual-panel figure: Accuracy (left) and Loss (right).

    A vertical green dashed line marks the Phase 1 → Phase 2 transition.
    An orange horizontal line marks the 88.23% previous SOTA benchmark.

    Args:
        exp_id: Experiment ID (1–5)
    """
    exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
    print(f"\nGenerating research plots for EXP{exp_id}: {exp_name}")

    # ── Load Phase 1 log ─────────────────────────────────────────────────────
    df_p1 = load_phase_log(exp_id, phase=1)
    if df_p1 is None:
        print(f"\n[ERROR] Cannot find logs/experiment_{exp_id}/training_log_phase1.csv")
        print(f"        Download it from Google Drive and place it in your local folder.")
        return

    print(f"  Phase 1 loaded: {len(df_p1)} epochs")

    # ── Load Phase 2 log (optional) ──────────────────────────────────────────
    df_p2 = load_phase_log(exp_id, phase=2)
    has_phase2 = df_p2 is not None
    if has_phase2:
        print(f"  Phase 2 loaded: {len(df_p2)} epochs")
        # Offset Phase 2 epoch numbers to continue from Phase 1
        phase2_offset = df_p1["epoch"].max()
        df_p2 = df_p2.copy()
        df_p2["epoch"] = df_p2["epoch"] + phase2_offset
    else:
        print("  Phase 2 log not found — plotting Phase 1 only.")

    # ── Combine histories ────────────────────────────────────────────────────
    if has_phase2:
        df_all = pd.concat([df_p1, df_p2], ignore_index=True)
        phase2_start_epoch = df_p1["epoch"].max() + 1
    else:
        df_all = df_p1
        phase2_start_epoch = None

    os.makedirs("results/figures", exist_ok=True)

    # ── Plot Setup ───────────────────────────────────────────────────────────
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"EXP{exp_id}: {exp_name}\n"
        f"Two-Phase Training: Warm-Up (Frozen) → Fine-Tuning (Unfrozen Backbone)",
        fontsize=13, fontweight="bold", y=1.02
    )

    epochs = df_all["epoch"]

    for ax, metric, val_metric, ylabel, title_suffix in [
        (axes[0], "accuracy",  "val_accuracy", "Accuracy",  "Accuracy Curve"),
        (axes[1], "loss",      "val_loss",      "Loss",      "Loss Curve"),
    ]:
        # Phase 1 region shading
        p1_end = df_p1["epoch"].max()
        ax.axvspan(0, p1_end, alpha=0.05, color="blue", label="_Phase 1 region")

        # Phase 2 region shading
        if has_phase2:
            ax.axvspan(p1_end, df_all["epoch"].max(), alpha=0.05, color="green", label="_Phase 2 region")

        # Training metric line
        ax.plot(epochs, df_all[metric],     color="#1f77b4", linewidth=2,
                label="Train (Mixup Phase 1 / Clean Phase 2)")

        # Validation metric line
        if val_metric in df_all.columns:
            ax.plot(epochs, df_all[val_metric], color="#d62728", linewidth=2,
                    linestyle="-", label="Validation (Clean)")

        # Phase 2 start vertical marker
        if phase2_start_epoch:
            ax.axvline(x=phase2_start_epoch, color="green", linestyle="--",
                       linewidth=1.5, label="Phase 2 Start (Backbone Unfrozen)")

        # SOTA benchmark line (accuracy plot only)
        if metric == "accuracy":
            ax.axhline(y=SOTA_ACCURACY, color="orange", linestyle=":",
                       linewidth=2, label=f"Previous SOTA ({SOTA_ACCURACY*100:.2f}%)")

        ax.set_title(f"EXP{exp_id} — {title_suffix}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.4)

    plt.tight_layout()

    save_path = f"results/figures/exp{exp_id}_two_phase_training.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n  [OK] High-resolution figure saved to: {save_path}")
    plt.show()

    # Print test results if available
    print_test_results(exp_id)


def generate_all_experiments() -> None:
    """Generate comparison summary plots for all completed experiments."""
    results = {}
    for exp_id in range(1, 6):
        results_path = Path(f"logs/experiment_{exp_id}/final_test_results.txt")
        if results_path.exists():
            text = results_path.read_text()
            for line in text.splitlines():
                if "Top-1 Accuracy" in line:
                    acc = float(line.split(":")[1].strip().replace("%", "")) / 100
                    results[exp_id] = acc

    if len(results) < 2:
        print("\n[INFO] Need at least 2 completed experiments to generate comparison.")
        return

    exp_labels = [EXP_NAMES.get(i, f"EXP{i}") for i in sorted(results.keys())]
    accuracies  = [results[i] for i in sorted(results.keys())]
    colors      = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(exp_labels, [a * 100 for a in accuracies],
                  color=colors[:len(results)], width=0.5, edgecolor="black")
    ax.axhline(y=SOTA_ACCURACY * 100, color="red", linestyle="--",
               linewidth=2, label=f"Previous SOTA: {SOTA_ACCURACY*100:.2f}%")

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc*100:.2f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=10)

    ax.set_title("SSL400 — All Experiments Accuracy Comparison\n"
                 "(Two-Phase Training: Frozen Warm-Up + Unfrozen Fine-Tuning)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Test Top-1 Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()

    save_path = "results/figures/all_experiments_comparison.png"
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n  [OK] Comparison chart saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SSL400 research plots")
    parser.add_argument("--exp_id", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Experiment ID to plot (default: 1)")
    parser.add_argument("--all", action="store_true",
                        help="Generate comparison chart for all completed experiments")
    args = parser.parse_args()

    if args.all:
        generate_all_experiments()
    else:
        generate_research_plots(exp_id=args.exp_id)
