import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_research_plots(exp_id=1):
    csv_path = f"logs/experiment_{exp_id}/training_log_phase1.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: Cannot find {csv_path}!")
        print("Please download it from Google Drive (ssl400_research_project/logs/experiment_1/) and put it in your local logs/experiment_1/ folder.")
        return

    # Load the training data saved by Colab
    df = pd.read_csv(csv_path)
    
    epochs = df['epoch'] + 1  # Epochs are 0-indexed in Keras
    
    # Create the beautiful dual-plot figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy
    axes[0].plot(epochs, df['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    if 'val_accuracy' in df.columns:
        axes[0].plot(epochs, df['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
    axes[0].set_title(f'SSL400 I3D Backbone - Accuracy (Exp {exp_id})', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend(fontsize=12)
    
    # Plot 2: Loss
    axes[1].plot(epochs, df['loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in df.columns:
        axes[1].plot(epochs, df['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[1].set_title(f'SSL400 I3D Backbone - Loss (Exp {exp_id})', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot as a high-quality image for the research paper
    save_path = f"results/experiment_{exp_id}_research_report.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    
    print(f"✅ Success! Beautiful research graphs generated and saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    generate_research_plots(exp_id=1)
