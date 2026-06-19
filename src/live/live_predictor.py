"""
live_predictor.py
=================
Phase 5 — Live Webcam Inference

Captures live webcam feed, applies the specified enhancement pipeline,
runs the trained I3D model continuously, and provides real-time predictions.
If a prediction stays confident for a set duration, the TTS module speaks
the prediction in Sinhala.

Usage:
    python src/live/live_predictor.py --exp_id 1
"""

import argparse
import collections
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from PIL import Image, ImageDraw, ImageFont

# ── Environment Setup ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from src.enhancement.enhancement_factory import get_enhancer
from src.live.text_to_speech import SinhalaTTS

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


def load_class_mapping() -> dict:
    """Loads class names from sinhala_word_map.csv."""
    try:
        df = pd.read_csv("data/splits/sinhala_word_map.csv")
        # Ensure we have the Sinhala word
        return dict(zip(df["class_id"], df["sinhala_word"]))
    except Exception as e:
        logger.error(f"Failed to load class mapping: {e}")
        return {}


def render_sinhala_text(img_np: np.ndarray, text: str, position: tuple, font_size: int = 40, color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    Renders Sinhala text on an OpenCV image using PIL.
    OpenCV's putText does not support Unicode / complex text rendering.
    """
    # Convert OpenCV BGR image to PIL RGB
    img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Try to load Windows Sinhala font (Iskoola Pota or Nirmala UI)
    font_path = "C:/Windows/Fonts/iskpota.ttf"
    if not os.path.exists(font_path):
        font_path = "C:/Windows/Fonts/Nirmala.ttf"
        
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
        
    draw.text(position, text, font=font, fill=color)
    
    # Convert back to OpenCV BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def run_live_inference(exp_id: int, config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    exp_config = config["experiments"][exp_id]
    
    model_path = Path(exp_config["model_dir"]) / "best_model.keras"
    if not model_path.exists():
        logger.error(f"Trained model not found for EXP{exp_id}: {model_path}")
        sys.exit(1)

    logger.info(f"Loading Model for EXP{exp_id}: {exp_config['name']} ...")
    model = tf.keras.models.load_model(str(model_path))
    
    class_map = load_class_mapping()
    num_classes = config["model"]["num_classes"]
    target_frames = config["video"]["target_frames"]
    spatial_size = tuple(config["model"]["input_shape"][1:3]) # (224, 224)
    
    # Load enhancement function (handles resizing and normalization)
    enhancer_fn = get_enhancer(exp_id, config_path)
    
    # Initialize TTS
    tts = SinhalaTTS()
    
    # Frame buffer
    frame_buffer = collections.deque(maxlen=target_frames)
    
    # Prediction state
    current_prediction = "Waiting for motion..."
    current_confidence = 0.0
    last_spoken_word = ""
    frames_stable = 0
    CONFIDENCE_THRESHOLD = 0.70
    STABILITY_FRAMES = 15  # Need ~0.5s of stable prediction to trigger TTS
    
    # ── OpenCV Video Capture ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam.")
        sys.exit(1)
        
    # Set webcam resolution to 720p for better display
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    logger.info("Webcam started. Press 'q' to quit.")
    
    # Processing rate limiter
    inference_interval = 3  # Run inference every 3 frames to save CPU
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Flip horizontally for mirror effect (more natural for signing)
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # 1. Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. Add to rolling buffer
            frame_buffer.append(rgb_frame)
            
            # 3. Run Inference if buffer is full and interval reached
            if len(frame_buffer) == target_frames and frame_count % inference_interval == 0:
                # Stack frames into video tensor (32, H, W, 3)
                video_clip = np.stack(list(frame_buffer))
                
                # Apply enhancement pipeline (outputs shape (32, 224, 224, 3), range [-1, 1])
                enhanced_clip = enhancer_fn(video_clip)
                
                # Add batch dimension: (1, 32, 224, 224, 3)
                batch_clip = tf.expand_dims(enhanced_clip, axis=0)
                
                # Predict
                preds = model.predict(batch_clip, verbose=0)[0]
                
                pred_idx = np.argmax(preds)
                confidence = preds[pred_idx]
                
                # State logic for stabilizing predictions
                if confidence > CONFIDENCE_THRESHOLD:
                    predicted_word = class_map.get(pred_idx, f"Class {pred_idx}")
                    
                    if predicted_word == current_prediction:
                        frames_stable += inference_interval
                    else:
                        current_prediction = predicted_word
                        frames_stable = 0
                        
                    current_confidence = confidence
                    
                    # Trigger TTS if stable and hasn't just been spoken
                    if frames_stable >= STABILITY_FRAMES and current_prediction != last_spoken_word:
                        logger.info(f"🗣️ Speaking: {current_prediction} ({confidence:.2f})")
                        tts.speak(current_prediction)
                        last_spoken_word = current_prediction
                        frames_stable = 0  # reset to prevent spamming
                else:
                    # Confidence too low
                    frames_stable = 0
            
            # ── Draw Overlay ──────────────────────────────────────────────────
            # Draw confidence bar
            bar_width = int(400 * current_confidence)
            color = (0, 255, 0) if current_confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.rectangle(display_frame, (50, 60), (50 + bar_width, 80), color, -1)
            cv2.rectangle(display_frame, (50, 60), (450, 80), (255, 255, 255), 2)
            
            # Display stats using standard OpenCV text
            cv2.putText(display_frame, f"Conf: {current_confidence*100:.1f}%", (470, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"EXP{exp_id}: {exp_config['name']}", (50, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            
            # Render Sinhala prediction text using PIL
            if current_prediction != "Waiting for motion...":
                display_frame = render_sinhala_text(
                    display_frame, 
                    f"Prediction: {current_prediction}", 
                    position=(50, 100), 
                    font_size=48, 
                    color=(0, 255, 0) if current_confidence > CONFIDENCE_THRESHOLD else (255, 255, 0)
                )
            else:
                cv2.putText(display_frame, current_prediction, (50, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
            
            # Show the frame
            cv2.imshow("SSL400 Live Predictor", display_frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        logger.info("Shutting down...")
        cap.release()
        cv2.destroyAllWindows()
        tts.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live WebCam Predictor for SSL400")
    parser.add_argument("--exp_id", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    run_live_inference(args.exp_id, args.config)
