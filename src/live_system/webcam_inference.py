"""
webcam_inference.py
===================
Real-time Sinhala Sign Language detection system using webcam input.

Pipeline:
    Webcam → Frame Buffer (60 frames @ 20 FPS) → Enhancement → Sample 32 frames
    → Normalize → Model Inference → Temporal Smoothing → Sentence Builder
    → Sinhala Text Rendering → Display

Usage:
    python src/live_system/webcam_inference.py --exp_id 5
    python src/live_system/webcam_inference.py --exp_id 5 --camera 0
"""

import argparse
import collections
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import yaml

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
try:
    import tf_keras as keras
except ImportError:
    keras = tf.keras

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_word_map(csv_path: str) -> dict:
    """Load class_id → sinhala_word mapping from CSV."""
    df = pd.read_csv(csv_path)
    return {int(row["class_id"]): row["class_name_sinhala"] for _, row in df.iterrows()}


class TemporalSmoother:
    """
    Reduces prediction flickering via a sliding window majority vote.

    Collects predictions over a window of N frames and returns a stable
    prediction only when one class achieves a 60% majority vote.
    """

    def __init__(self, window_size: int = 5, confidence_threshold: float = 0.65):
        self.window_size = window_size
        self.threshold = confidence_threshold
        self.prediction_window = collections.deque(maxlen=window_size)

    def update(self, class_id: int, confidence: float) -> Optional[int]:
        """
        Update the window with a new prediction and return a stable class if majority found.

        Args:
            class_id:   Predicted class index.
            confidence: Softmax probability of the predicted class.

        Returns:
            Stable class ID if majority vote achieved, else None.
        """
        if confidence >= self.threshold:
            self.prediction_window.append(class_id)

        if len(self.prediction_window) == self.window_size:
            counts = collections.Counter(self.prediction_window)
            most_common_class, count = counts.most_common(1)[0]
            if count >= int(self.window_size * 0.6):
                return most_common_class
        return None


class SentenceBuilder:
    """Accumulates confirmed sign predictions into a readable sentence."""

    def __init__(self, max_words: int = 10, reset_timeout_sec: float = 3.0):
        self.words = []
        self.last_word = None
        self.last_word_time = 0.0
        self.reset_timeout = reset_timeout_sec
        self.max_words = max_words

    def add_word(self, sinhala_word: str) -> str:
        """
        Add a word to the sentence, suppressing duplicates and auto-resetting on pause.

        Args:
            sinhala_word: The Sinhala Unicode word to add.

        Returns:
            The current sentence as a space-joined string.
        """
        now = time.time()
        if now - self.last_word_time > self.reset_timeout:
            self.words = []

        if sinhala_word != self.last_word:
            self.words.append(sinhala_word)
            self.last_word = sinhala_word
            self.last_word_time = now

        if len(self.words) > self.max_words:
            self.words.pop(0)

        return " ".join(self.words)

    def clear(self) -> None:
        """Reset the sentence."""
        self.words = []
        self.last_word = None


def render_sinhala_text(
    frame_bgr: np.ndarray,
    sinhala_text: str,
    font_path: str,
    font_size: int = 28,
    position: tuple = (10, 10)
) -> np.ndarray:
    """
    Render Sinhala Unicode text onto an OpenCV BGR frame using PIL.

    OpenCV does not natively support Unicode/Sinhala script rendering.
    We convert to PIL, draw text with the Iskoola Pota font, then convert back.

    Args:
        frame_bgr:   OpenCV BGR frame (numpy uint8).
        sinhala_text: Sinhala Unicode string to render.
        font_path:   Path to .ttf font file (Iskoola Pota recommended).
        font_size:   Font size in points.
        position:    (x, y) pixel position for text.

    Returns:
        Frame with text rendered (BGR uint8).
    """
    from PIL import Image, ImageDraw, ImageFont

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        logger.warning("Sinhala font not found. Using default font.")
        font = ImageFont.load_default()

    # Draw semi-transparent background for readability
    try:
        text_bbox = draw.textbbox(position, sinhala_text, font=font)
        draw.rectangle(
            [text_bbox[0] - 5, text_bbox[1] - 5,
             text_bbox[2] + 5, text_bbox[3] + 5],
            fill=(0, 0, 0, 180)
        )
    except Exception:
        pass  # Older PIL versions may not have textbbox

    draw.text(position, sinhala_text, font=font, fill=(255, 255, 0))

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def preprocess_buffer(
    frame_buffer: list,
    enhance_fn,
    num_frames: int = 32,
    target_size: tuple = (224, 224)
) -> np.ndarray:
    """
    Process a frame buffer into a model-ready tensor.

    Args:
        frame_buffer:  List of raw BGR numpy frames.
        enhance_fn:    Enhancement function for the winning experiment.
        num_frames:    Number of frames to sample.
        target_size:   (width, height) for resizing.

    Returns:
        Float32 numpy array, shape (1, num_frames, 224, 224, 3) normalized to [-1, 1].
    """
    total = len(frame_buffer)
    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    processed = []

    for idx in indices:
        frame = frame_buffer[idx].copy()
        frame = enhance_fn(frame)
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)
        frame = (frame / 127.5) - 1.0
        processed.append(frame)

    tensor = np.stack(processed, axis=0)
    return tensor[np.newaxis, ...]  # Add batch dimension: (1, 32, 224, 224, 3)


def run_live_detection(
    model: keras.Model,
    enhance_fn,
    word_map: dict,
    config: dict,
    camera_id: int = 0
) -> None:
    """
    Main real-time detection loop.

    Reads frames from webcam, accumulates a 3-second buffer, runs inference,
    and displays the Sinhala prediction on screen.

    Controls:
        'q' or 'ESC' → Quit
        'c'          → Clear sentence

    Args:
        model:      Loaded Keras model.
        enhance_fn: Frame enhancement function.
        word_map:   class_id → Sinhala word mapping.
        config:     Loaded config.yaml.
        camera_id:  OpenCV camera index (0 = default webcam).
    """
    live_cfg = config["live"]
    fps = config["dataset"]["fps"]
    clip_duration = config["dataset"]["clip_duration_sec"]
    buffer_size = fps * clip_duration  # 20 * 3 = 60 frames
    num_frames = config["frames"]["num_frames"]
    font_path = str(PROJECT_ROOT / live_cfg["font_path"])

    smoother = TemporalSmoother(
        window_size=live_cfg["temporal_window_size"],
        confidence_threshold=live_cfg["confidence_threshold"]
    )
    sentence_builder = SentenceBuilder(
        max_words=live_cfg["sentence_max_words"],
        reset_timeout_sec=live_cfg["sentence_reset_timeout_sec"]
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_id}")
        return

    logger.info("✅ Live detection started. Press 'q' to quit, 'c' to clear sentence.")

    frame_buffer = []
    current_sentence = ""
    current_prediction = ""
    current_confidence = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame capture failed.")
            break

        frame_buffer.append(frame.copy())

        # Process every 60 frames (3-second buffer)
        if len(frame_buffer) >= buffer_size:
            tensor = preprocess_buffer(frame_buffer, enhance_fn, num_frames)
            frame_buffer = []

            # Model inference
            logits = model.predict(tensor, verbose=0)[0]
            probs = tf.nn.softmax(logits).numpy()
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])

            current_confidence = confidence
            current_prediction = word_map.get(class_id, f"Class_{class_id}")

            # Temporal smoothing
            stable_class = smoother.update(class_id, confidence)
            if stable_class is not None:
                sinhala_word = word_map.get(stable_class, f"Class_{stable_class}")
                current_sentence = sentence_builder.add_word(sinhala_word)

        # Draw overlay on display frame
        display = frame.copy()

        # Current prediction bar
        cv2.rectangle(display, (0, 0), (640, 70), (0, 0, 0), -1)
        if current_confidence > 0:
            bar_width = int(current_confidence * 600)
            color = (0, 255, 0) if current_confidence > 0.65 else (0, 165, 255)
            cv2.rectangle(display, (10, 10), (10 + bar_width, 30), color, -1)
            cv2.putText(display, f"Confidence: {current_confidence*100:.1f}%",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Sinhala sentence display (bottom of screen)
        if current_sentence:
            display = render_sinhala_text(
                display, current_sentence, font_path,
                font_size=live_cfg["font_size"],
                position=(10, display.shape[0] - 60)
            )

        # Frame count progress bar
        progress = len(frame_buffer) / buffer_size
        cv2.rectangle(display, (0, display.shape[0] - 10),
                      (int(progress * display.shape[1]), display.shape[0]),
                      (100, 200, 255), -1)

        cv2.imshow("SSL400 Sinhala Sign Language Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), 27]:  # q or ESC
            break
        elif key == ord("c"):
            sentence_builder.clear()
            current_sentence = ""
            logger.info("Sentence cleared.")

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Live detection stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time Sinhala Sign Language detection.")
    parser.add_argument("--exp_id", type=int, default=5, choices=[1, 2, 3, 4, 5],
                        help="Experiment ID for the model to use (default: 5 = best hybrid)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    args = parser.parse_args()

    config = load_config()
    exp = [e for e in config["experiments"] if e["id"] == args.exp_id][0]

    # Load model
    model_path = PROJECT_ROOT / exp["model_dir"] / "best_model_phase2.keras"
    if not model_path.exists():
        model_path = PROJECT_ROOT / exp["model_dir"] / "best_model_phase1.keras"
    logger.info(f"Loading model: {model_path}")
    model = keras.models.load_model(str(model_path))

    # Load enhancement function
    from enhancement.enhancement_factory import get_enhancer
    enhance_fn = get_enhancer(args.exp_id)

    # Load word map
    word_map_path = str(PROJECT_ROOT / config["paths"]["splits"] / "sinhala_word_map.csv")
    word_map = load_word_map(word_map_path)

    run_live_detection(model, enhance_fn, word_map, config, camera_id=args.camera)


if __name__ == "__main__":
    main()
