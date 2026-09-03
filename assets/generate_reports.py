import os
import matplotlib.pyplot as plt
import numpy as np

# Data definitions for all 4 experiments
experiments = [
    {
        "id": 1,
        "title": "EXP 1: Baseline (No Enhancement)",
        "file_name": "EXP1_Baseline_Performance.png",
        "cm": np.array([
            [17, 0, 0, 0, 1, 0, 1, 0],
            [0, 18, 0, 0, 0, 0, 0, 0],
            [1, 0, 16, 0, 0, 0, 2, 0],
            [0, 0, 0, 16, 0, 0, 0, 0],
            [1, 0, 0, 0, 13, 0, 1, 0],
            [0, 1, 0, 0, 0, 12, 0, 0],
            [4, 0, 2, 0, 0, 0, 7, 0],
            [0, 0, 2, 0, 0, 0, 0, 11]
        ]),
        "metrics": [
            ["Accuracy", "0.83"],
            ["Macro Precision", "0.84"],
            ["Macro Recall", "0.80"],
            ["Macro F1-score", "0.81"],
            ["Test Samples", "126"],
            ["Correct Predictions", "104"]
        ]
    },
    {
        "id": 2,
        "title": "EXP 2: CLAHE + Gamma Correction",
        "file_name": "EXP2_CLAHE_Gamma_Performance.png",
        "cm": np.array([
            [19, 0, 0, 0, 0, 0, 0, 0],
            [0, 18, 0, 0, 0, 0, 0, 0],
            [0, 1, 18, 0, 0, 0, 0, 0],
            [0, 0, 0, 16, 0, 0, 0, 0],
            [1, 0, 1, 0, 13, 0, 0, 0],
            [0, 0, 0, 0, 0, 13, 0, 0],
            [0, 0, 2, 0, 0, 0, 11, 0],
            [0, 0, 2, 0, 0, 0, 1, 10]
        ]),
        "metrics": [
            ["Accuracy", "0.94"],
            ["Macro Precision", "0.95"],
            ["Macro Recall", "0.93"],
            ["Macro F1-score", "0.94"],
            ["Test Samples", "126"],
            ["Correct Predictions", "118"]
        ]
    },
    {
        "id": 3,
        "title": "EXP 3: Edge-Preserving Sharpening (Bilateral + Unsharp)",
        "file_name": "EXP3_Bilateral_Unsharp_Performance.png",
        "cm": np.array([
            [16, 0, 0, 0, 1, 0, 2, 0],
            [0, 18, 0, 0, 0, 0, 0, 0],
            [0, 0, 18, 0, 0, 0, 1, 0],
            [0, 0, 1, 15, 0, 0, 0, 0],
            [0, 0, 1, 0, 13, 0, 1, 0],
            [0, 0, 0, 0, 1, 12, 0, 0],
            [1, 0, 1, 0, 0, 0, 11, 0],
            [0, 0, 2, 0, 0, 0, 0, 11]
        ]),
        "metrics": [
            ["Accuracy", "0.90"],
            ["Macro Precision", "0.92"],
            ["Macro Recall", "0.90"],
            ["Macro F1-score", "0.91"],
            ["Test Samples", "126"],
            ["Correct Predictions", "114"]
        ]
    },
    {
        "id": 4,
        "title": "EXP 4: Hybrid (Bilateral + CLAHE + Unsharp)",
        "file_name": "EXP4_Hybrid_Performance.png",
        "cm": np.array([
            [15, 0, 0, 0, 1, 0, 3, 0],
            [0, 17, 1, 0, 0, 0, 0, 0],
            [0, 0, 16, 0, 0, 0, 3, 0],
            [0, 0, 0, 16, 0, 0, 0, 0],
            [2, 0, 1, 0, 12, 0, 0, 0],
            [0, 1, 0, 0, 0, 12, 0, 0],
            [1, 0, 1, 0, 1, 0, 10, 0],
            [0, 0, 2, 0, 0, 0, 0, 11]
        ]),
        "metrics": [
            ["Accuracy", "0.87"],
            ["Macro Precision", "0.88"],
            ["Macro Recall", "0.86"],
            ["Macro F1-score", "0.87"],
            ["Test Samples", "126"],
            ["Correct Predictions", "109"]
        ]
    }
]

labels = ['Thank you', 'Hello', 'Good', 'House', 'Eat', 'Drink', 'Tell', 'Write']
out_dir = r"c:\project\ssl400_research_project\assets\experiment_reports"
os.makedirs(out_dir, exist_ok=True)

for exp in experiments:
    fig = plt.figure(figsize=(13, 7.5), dpi=300, facecolor='white')
    
    # Overall Title
    fig.suptitle(exp["title"], fontsize=18, fontweight='bold', y=0.96)
    
    # Grid: 1 row, 2 columns (Left: Confusion Matrix, Right: Performance Table)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 0.7], wspace=0.3, left=0.08, right=0.95, top=0.86, bottom=0.15)
    
    # --- LEFT: Confusion Matrix ---
    ax_cm = fig.add_subplot(gs[0, 0])
    ax_cm.set_title("Confusion Matrix", fontsize=14, fontweight='bold', pad=12)
    
    cm = exp["cm"]
    im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=19)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.set_label("Number of Samples", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    # Ticks and Labels
    tick_marks = np.arange(len(labels))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_yticklabels(labels, fontsize=10)
    
    ax_cm.set_ylabel("True Label", fontsize=11, labelpad=8)
    ax_cm.set_xlabel("Predicted Label", fontsize=11, labelpad=12)
    
    # Text annotations in matrix cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax_cm.text(j, i, format(val, 'd'),
                       ha="center", va="center",
                       color=color, fontsize=10)
            
    # --- RIGHT: Overall Performance Table ---
    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.axis('off')
    ax_tbl.set_title("Overall Performance", fontsize=14, fontweight='bold', pad=12)
    
    table_data = [["Metric", "Score"]] + exp["metrics"]
    
    # Draw table
    tbl = ax_tbl.table(
        cellText=table_data,
        loc='center',
        cellLoc='left',
        colWidths=[0.55, 0.45]
    )
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 2.2)
    
    # Style cells
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1.0)
        if r == 0:
            cell.set_text_props(weight='bold')
        elif c == 1:
            cell.set_text_props(weight='bold')
            
    # Save image
    out_path = os.path.join(out_dir, exp["file_name"])
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {out_path}")

print("All 4 experiment report images generated successfully!")
