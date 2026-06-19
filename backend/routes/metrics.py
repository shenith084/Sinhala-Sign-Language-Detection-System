from flask import Blueprint, jsonify
from backend.services.model_service import get_model

metrics_bp = Blueprint("metrics", __name__)

@metrics_bp.route("/status", methods=["GET"])
def status():
    """Health check and model status."""
    return jsonify({
        "status": "online",
        "model_loaded": get_model() is not None
    })
