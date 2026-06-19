import argparse
import os
import sys
import yaml
from pathlib import Path
from flask import Flask
from flask_cors import CORS

# ── Environment Setup ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from backend.utils.logger import get_logger
from backend.utils.sinhala_dictionary import load_dictionary
from backend.services.model_service import load_model
from backend.services.enhancement_service import init_enhancer
from backend.routes.predict import predict_bp
from backend.routes.metrics import metrics_bp
from backend.routes.experiments import experiments_bp

logger = get_logger(__name__)

def create_app(exp_id: int, config_path: str = "config.yaml"):
    app = Flask(__name__)
    CORS(app)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Initializing Backend System...")
    load_dictionary()
    init_enhancer(exp_id, config_path)
    load_model(exp_id, config)

    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(metrics_bp, url_prefix="/api/metrics")
    app.register_blueprint(experiments_bp, url_prefix="/api/experiments")

    logger.info("Backend is fully initialized and ready.")
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSL400 Flask Backend API")
    parser.add_argument("--exp_id", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app = create_app(args.exp_id, args.config)
    app.run(host="0.0.0.0", port=args.port, debug=False)
