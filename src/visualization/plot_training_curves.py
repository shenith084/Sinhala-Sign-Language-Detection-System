"""
plot_training_curves.py
=======================
Generate beautiful thesis-ready Accuracy and Loss graphs from CSV logs.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def plot_experiment(exp_id: int):
    with open(PROJECT_ROOT / 'config.yaml') as f:
        config = yaml.safe_load(f)
    
    exp = [e for e in config['experiments'] if e['id'] == exp_id][0]
    log_dir = PROJECT_ROOT / exp['log_dir']
    
    # Try Phase 2 first, fallback to Phase 1
    log_file = log_dir / 'training_log_phase2.csv'
    if not log_file.exists():
        log_file = log_dir / 'training_log_phase1.csv'
        
    if not log_file.exists():
        print(f"❌ No logs found for Experiment {exp_id}")
        return

    df = pd.read_csv(log_file)
    
    # Set up matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Accuracy
    ax1.plot(df['accuracy'], label='Training Accuracy', linewidth=2, color='#2ca02c')
    ax1.plot(df['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#d62728', linestyle='--')
    ax1.set_title(f'Accuracy Curve - {exp["name"]}', fontsize=14, pad=15)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=11)
    
    # Plot Loss
    ax2.plot(df['loss'], label='Training Loss', linewidth=2, color='#1f77b4')
    ax2.plot(df['val_loss'], label='Validation Loss', linewidth=2, color='#ff7f0e', linestyle='--')
    ax2.set_title(f'Loss Curve - {exp["name"]}', fontsize=14, pad=15)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save Figure
    out_dir = PROJECT_ROOT / config['paths']['results'] / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'exp{exp_id}_training_curves.png'
    
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Success! Saved beautiful thesis graph to {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, required=True)
    args = parser.parse_args()
    plot_experiment(args.exp_id)
