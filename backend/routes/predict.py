from flask import Blueprint, request, jsonify
import base64
import cv2
import numpy as np

from services.model_service import ModelService
from services.enhancement_service import EnhancementService
from utils.logger import logger
from utils.sinhala_dictionary import SinhalaDictionary
from pathlib import Path

predict_bp = Blueprint('predict_bp', __name__)

model_service = ModelService()
enhancement_service = EnhancementService()

# Load Sinhala dictionary
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
dict_path = PROJECT_ROOT / "data" / "splits" / "sinhala_word_map.csv"
sinhala_dict = SinhalaDictionary(str(dict_path))

@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'frames' not in data:
            return jsonify({"error": "No frames provided"}), 400

        exp_id = int(data.get('exp_id', 1))  # Default to experiment 1
        frames_b64 = data['frames']
        
        logger.info(f"Received prediction request for exp_id: {exp_id}, active_exp_id: {model_service.active_exp_id}")
        
        # Ensure correct model is loaded
        if model_service.active_exp_id != exp_id:
            logger.info("Active exp_id doesn't match! Reloading model...")
            if exp_id == 2:
                model_path = PROJECT_ROOT / "models" / f"experiment_{exp_id}" / "best_model_phase2.keras"
            else:
                model_path = PROJECT_ROOT / "models" / f"experiment_{exp_id}" / "saved_model"
            
            if not model_service.load_model(exp_id, str(model_path)):
                logger.error("load_model returned False!")
                return jsonify({"error": f"Failed to load model for experiment {exp_id}"}), 500
        else:
            logger.info("Model already loaded. Bypassing load_model().")

        # Decode base64 frames
        raw_frames = []
        for b64_str in frames_b64:
            # Remove data URI prefix if present
            if ',' in b64_str:
                b64_str = b64_str.split(',')[1]
            
            img_data = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                raw_frames.append(frame)

        if len(raw_frames) < 5:
            return jsonify({"error": "Too few frames provided"}), 400

        # Preprocess
        tensor = enhancement_service.preprocess_frames(raw_frames, exp_id)
        
        # Predict using real model
        probs = model_service.predict(tensor)
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        logger.info(f"Probs: {probs}, class_id: {class_id}, confidence: {confidence}")
        
        # Map class_id to words
        word_info = sinhala_dict.get_word(class_id)
        
        return jsonify({
            "class_id": class_id,
            "confidence": confidence,
            "word_english": word_info['english'],
            "word_sinhala": word_info['sinhala']
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
