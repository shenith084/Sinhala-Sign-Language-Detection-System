"""
statistical_analysis.py
=======================
Phase 4 — Cross-Experiment Analysis

Aggregates the evaluation metrics from all 5 experiments, generates
a summary comparison table, and plots bar charts comparing Accuracy
and Macro F1-Score across the different enhancement techniques.

Usage:
    python src/evaluation/statistical_analysis.py
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

# ── Environment Setup ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["PYTHONIOENCODING"] = "utf-8"

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


def run_analysis() -> None:
    config = load_config("config.yaml")
    
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # ── 1. Aggregate Metrics ──────────────────────────────────────────────────
    all_metrics = []
    for exp_id in range(1, 6):
        exp_csv = results_dir / f"experiment_{exp_id}" / "summary_metrics.csv"
        if exp_csv.exists():
            df = pd.read_csv(exp_csv)
            all_metrics.append(df)
        else:
            logger.warning(f"Metrics not found for EXP{exp_id}: {exp_csv}")
            
    if not all_metrics:
        logger.error("No experiment metrics found. Run evaluate.py for experiments first.")
        sys.exit(1)
        
    combined_df = pd.concat(all_metrics, ignore_index=True)
    
    # Save combined table
    combined_csv = results_dir / "all_experiments_summary.csv"
    combined_df.to_csv(combined_csv, index=False)
    logger.info(f"Saved combined summary table -> {combined_csv}")
    
    # Log formatted table to console
    logger.info("\n" + combined_df.to_string(index=False))
    
    # ── 2. Plot Comparisons ───────────────────────────────────────────────────
    sns.set_theme(style="whitegrid")
    
    # Accuracy Comparison
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=combined_df, 
        x="Enhancement", 
        y="Accuracy", 
        hue="Enhancement",
        palette="viridis",
        legend=False
    )
    plt.title("Test Accuracy Comparison by Enhancement Technique", fontsize=14)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Enhancement Pipeline", fontsize=12)
    plt.ylim(0, 1.0)
    
    # Add value labels
    for i, v in enumerate(combined_df["Accuracy"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')
        
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    acc_plot_path = figures_dir / "accuracy_comparison.png"
    plt.savefig(acc_plot_path, dpi=150)
    plt.close()
    
    # F1-Score Comparison
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=combined_df, 
        x="Enhancement", 
        y="Macro_F1", 
        hue="Enhancement",
        palette="magma",
        legend=False
    )
    plt.title("Macro F1-Score Comparison by Enhancement Technique", fontsize=14)
    plt.ylabel("Macro F1-Score", fontsize=12)
    plt.xlabel("Enhancement Pipeline", fontsize=12)
    plt.ylim(0, 1.0)
    
    # Add value labels
    for i, v in enumerate(combined_df["Macro_F1"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')
        
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    f1_plot_path = figures_dir / "f1_score_comparison.png"
    plt.savefig(f1_plot_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved accuracy plot -> {acc_plot_path}")
    logger.info(f"Saved F1-score plot -> {f1_plot_path}")
    logger.info("Cross-experiment analysis complete.")

if __name__ == "__main__":
    run_analysis()
