import base64
import tempfile
import cv2
import numpy as np
import tensorflow as tf
from flask import Blueprint, request, jsonify, send_file
from gtts import gTTS

from backend.services.model_service import get_model, predict_batch, get_target_frames
from backend.services.enhancement_service import apply_enhancement
from backend.utils.sinhala_dictionary import get_word
from backend.utils.logger import get_logger

logger = get_logger(__name__)
predict_bp = Blueprint("predict", __name__)

@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    Receives base64 encoded frames, applies enhancement, runs prediction,
    and returns top 3 classes.
    """
    if get_model() is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.json
    if not data or "frames" not in data:
        return jsonify({"error": "No frames provided"}), 400
        
    b64_frames = data["frames"]
    if len(b64_frames) == 0:
        return jsonify({"error": "Empty frame list"}), 400
        
    decoded_frames = []
    try:
        for b64 in b64_frames:
            if "," in b64:
                b64 = b64.split(",")[1]
            img_data = base64.b64decode(b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                decoded_frames.append(rgb_frame)
    except Exception as e:
        logger.error(f"Error decoding frames: {e}")
        return jsonify({"error": "Failed to decode frames"}), 400
        
    if len(decoded_frames) < 10:
        return jsonify({"error": "Not enough valid frames"}), 400
        
    # Resample
    target_f = get_target_frames()
    indices = np.linspace(0, len(decoded_frames) - 1, target_f).astype(int)
    video_clip = np.stack([decoded_frames[i] for i in indices])
    
    # Enhance & Predict
    enhanced_clip = apply_enhancement(video_clip)
    batch_clip = tf.expand_dims(enhanced_clip, axis=0)
    preds = predict_batch(batch_clip)
    
    # Top 3
    top3_idx = np.argsort(preds)[-3:][::-1]
    results = [
        {
            "class_id": int(idx),
            "sinhala_word": get_word(idx),
            "confidence": float(preds[idx])
        }
        for idx in top3_idx
    ]
        
    return jsonify({"success": True, "predictions": results})

@predict_bp.route("/tts", methods=["GET"])
def text_to_speech():
    """Returns MP3 TTS for the given Sinhala text."""
    text = request.args.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    try:
        tts = gTTS(text=text, lang="si", slow=False)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return send_file(temp_file.name, mimetype="audio/mpeg", as_attachment=True, download_name="prediction.mp3")
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return jsonify({"error": "TTS failed"}), 500
