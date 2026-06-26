import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from flask import Flask, jsonify
from flask_cors import CORS

from routes.predict import predict_bp
from routes.metrics import metrics_bp
from routes.experiments import experiments_bp
from services.model_service import ModelService
from utils.logger import logger
from pathlib import Path

app = Flask(__name__)

# Allow cross-origin requests from the React frontend
CORS(app, resources={r"/*": {"origins": "*"}})

# Register blueprints
app.register_blueprint(predict_bp, url_prefix='/api')
app.register_blueprint(metrics_bp, url_prefix='/api')
app.register_blueprint(experiments_bp, url_prefix='/api')

@app.route('/api/health', methods=['GET'])
def health_check():
    model_service = ModelService()
    return jsonify({
        "status": "ok",
        "model_loaded": model_service.model is not None,
        "active_experiment": model_service.active_exp_id
    })

if __name__ == "__main__":
    logger.info("Starting SSL400 Backend API...")
    
    # Try to load best model (Exp 1) on startup to reduce initial latency
    try:
        model_service = ModelService()
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        best_model_path = PROJECT_ROOT / "models" / "experiment_1" / "saved_model"
        if best_model_path.exists():
            model_service.load_model(1, str(best_model_path))
        else:
            logger.info("No trained models found on startup. API ready, but model load will happen on first request if available.")
    except Exception as e:
        logger.warning(f"Startup model load failed (normal if no models trained yet): {e}")

    # Run app
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)
