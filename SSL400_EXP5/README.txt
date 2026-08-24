SSL400 EXP5 - EfficientNetV2-S + BiLSTM + CLAHE
=================================================

THIS ZIP CONTAINS EVERYTHING (CODE + RAW VIDEOS + SPLITS)

SETUP STEPS
-----------
1. Extract this entire zip into your Google Drive at:
   My Drive / SSL400_EXP5 /

2. Open SSL400_EXP5_EfficientNetV2_BiLSTM.ipynb in Google Colab

3. Run all cells in order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8

TRAINING TIME
-------------
Cell 4 (video processing): 30-60 minutes (generates .npy files)
Cell 7 (training):         4-5 hours
Cell 8 (evaluation):       5 minutes

SETTINGS
--------
- 32 frames per video
- BATCH_SIZE = 2 (required for EfficientNetV2 on Tesla T4)
- CLAHE + Gamma enhancement
- Model auto-saved to Drive every 5 epochs
