from flask import Blueprint, jsonify
import yaml

experiments_bp = Blueprint("experiments", __name__)

@experiments_bp.route("/", methods=["GET"])
def list_experiments():
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return jsonify({"success": True, "experiments": config.get("experiments", {})})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
