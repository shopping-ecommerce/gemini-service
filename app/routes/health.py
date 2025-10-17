from flask import Blueprint, jsonify, current_app
from datetime import datetime

health_bp = Blueprint("health_bp", __name__)

@health_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "mongodb": hasattr(current_app, "mongo_client"),
        "vertex_ai": hasattr(current_app, "embedding_model"),
        "timestamp": datetime.now().isoformat()
    })
