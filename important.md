# Important Context, Bugs, and Fixes

This document records the critical bugs discovered during the training of the SSL400 MoViNet-A2 model on Google Colab, the fixes applied, and important context for future experiments.


## 3. The Augmentation Bug (Accuracy capped at 25%)
**The Problem:** The `val_accuracy` was hard-capped at around 25%. Analysis showed that extreme spatial augmentations (Gaussian noise, massive hue shifts, and saturation jitter) were hardcoded in `augmentation.py`. These augmentations were completely scrambling the visual features of the hands.
**The Fix:** Removed the aggressive noise and color shifts from `src/training/augmentation.py`. The immediate result was a massive drop in the initial Epoch 1 Validation Loss from `2.29` down to `2.19`.

## 4. The Underfitting Plateau (Phase 2 gets stuck at ~30%)
**The Problem:** By Epoch 40 in Phase 2, the model plateaus at around `~30%` validation accuracy and `~86%` Top-5 accuracy. The training loss gets stuck at `1.92` and validation loss at `2.04`.
**The Context:**
*   **Tiny Validation Set:** There are only 44 validation videos. This means a single correct guess increases accuracy by `2.27%`. The accuracy jumps around simply due to high variance on a tiny sample size.
*   **Underfitting:** Because the training loss (`1.92`) and validation loss (`2.04`) are so close, the model is **underfitting**. It is not memorizing the training data.
*   **Frozen Layers & High Dropout:** The lower blocks of MoViNet (`stem`, `block0`, `block1`) are completely frozen, and dropout is set high to `40%`. 
*   **Learning Rate Decay:** The `CosineDecay` scheduler drops the learning rate from `1e-4` to near-zero (`~0.000005`) by Epoch 43, preventing the network from making any further progress.

**Future Fix (If higher accuracy is needed):**
To force the model to learn the small dataset more aggressively in future experiments:
1.  **Unfreeze the entire model:** Allow the lower blocks to train so it has more capacity.
2.  **Reduce Dropout:** Change `dropout_rate` in `config.yaml` from `0.4` to `0.2`.
3.  **Increase Learning Rate:** Change Phase 2 LR from `1e-4` to `5e-4` to give it more power to escape the plateau.
