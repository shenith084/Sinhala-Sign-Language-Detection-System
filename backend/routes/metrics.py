from flask import Blueprint, jsonify
import json
from pathlib import Path
from utils.logger import logger

metrics_bp = Blueprint('metrics_bp', __name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

@metrics_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """Retrieve evaluation metrics for all experiments."""
    try:
        metrics = []
        for exp_id in range(1, 6):
            json_file = RESULTS_DIR / f"experiment_{exp_id}_metrics.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    metrics.append(data)
            else:
                # Placeholder if not trained yet
                metrics.append({
                    "exp_id": exp_id,
                    "exp_name": f"Experiment {exp_id}",
                    "top1_accuracy": 0.0,
                    "macro_f1": 0.0,
                    "status": "Not Trained"
                })
                
        return jsonify({"experiments": metrics})

    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return jsonify({"error": str(e)}), 500
