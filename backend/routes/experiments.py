from flask import Blueprint, jsonify
import yaml
from pathlib import Path

experiments_bp = Blueprint('experiments_bp', __name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

@experiments_bp.route('/experiments', methods=['GET'])
def get_experiments():
    """Returns configuration details for all experiments."""
    try:
        config_path = PROJECT_ROOT / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        experiments = config.get('experiments', [])
        
        # Check model existence
        for exp in experiments:
            exp_id = exp['id']
            p2_exists = (PROJECT_ROOT / f"models/experiment_{exp_id}/best_model_phase2.keras").exists()
            p1_exists = (PROJECT_ROOT / f"models/experiment_{exp_id}/best_model_phase1.keras").exists()
            
            if p2_exists:
                exp['status'] = "Phase 2 Complete"
            elif p1_exists:
                exp['status'] = "Phase 1 Complete"
            else:
                exp['status'] = "Not Trained"
                
        return jsonify({"experiments": experiments})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
