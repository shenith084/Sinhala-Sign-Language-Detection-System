@echo off
echo Starting Data Processing for Experiment 1 (Baseline)...
python src\data\video_to_frames.py --exp_id 1

echo Starting Data Processing for Experiment 2 (CLAHE)...
python src\data\video_to_frames.py --exp_id 2

echo Starting Data Processing for Experiment 3 (Bilateral)...
python src\data\video_to_frames.py --exp_id 3

echo Starting Data Processing for Experiment 4 (Hybrid)...
python src\data\video_to_frames.py --exp_id 4

echo Zipping all processed data for Kaggle...
powershell -Command "Compress-Archive -Path 'data\processed' -DestinationPath 'ssl400_kaggle_data.zip' -Force"

echo DONE! Data is ready for Kaggle!
