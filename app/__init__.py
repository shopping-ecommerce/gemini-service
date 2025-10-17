import os
import logging
from flask import Flask
from .services.vertex_ai_service import VertexAIService
from .services.mongodb_service import MongoDBService
from .services.event_service import EventService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config=None):
    app = Flask(__name__)
    
    if config:
        app.config.update(config)
    
    # GCP configs
    PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
    LOCATION = os.environ.get("GCP_LOCATION", "asia-southeast1")
    INDEX_ENDPOINT_ID = os.environ.get("INDEX_ENDPOINT_ID")
    DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID")
    INDEX_ID = os.environ.get("INDEX_ID")
    
    # MongoDB config
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://root:root@localhost:27017/product_db?authSource=admin")
    
    # Validate
    if not PROJECT_ID:
        raise ValueError("GCP_PROJECT_ID is required")
    if not INDEX_ENDPOINT_ID:
        raise ValueError("INDEX_ENDPOINT_ID is required")
    
    logger.info(f"Initializing app with project: {PROJECT_ID}, location: {LOCATION}")
    
    # 1. Khởi tạo MongoDB Service
    try:
        mongodb_service = MongoDBService(uri=MONGODB_URI)
        app.config['MONGODB_SERVICE'] = mongodb_service
        logger.info("✓ MongoDB Service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB: {e}")
        raise
    
    # 2. Khởi tạo Vertex AI Service
    try:
        vertex_ai_service = VertexAIService(
            project_id=PROJECT_ID,
            location=LOCATION,
            index_endpoint_id=INDEX_ENDPOINT_ID,
            deployed_index_id=DEPLOYED_INDEX_ID,
            index_id=INDEX_ID
        )
        app.config['VERTEX_AI_SERVICE'] = vertex_ai_service
        logger.info("✓ Vertex AI Service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI: {e}")
        raise
    
    # 3. Khởi tạo Event Service
    event_service = EventService(mongodb_service=mongodb_service)
    app.config['EVENT_SERVICE'] = event_service
    logger.info("✓ Event Service initialized")
    
    # 4. Register blueprints với prefix /gemini
    try:
        from .routes.index_ops import index_bp
        from .routes.health import health_bp
        from .routes.products import products_bp
        from .routes.search import search_bp
        from .routes.events import events_bp
        from .routes.similar import similar_bp
        from .routes.recommend import recommend_bp

        # Tất cả routes đều dưới /gemini
        app.register_blueprint(health_bp,     url_prefix="/gemini")
        app.register_blueprint(index_bp,      url_prefix="/gemini/index")
        app.register_blueprint(search_bp,     url_prefix="/gemini/search")
        app.register_blueprint(products_bp,   url_prefix="/gemini/products")
        app.register_blueprint(events_bp,     url_prefix="/gemini/events")
        app.register_blueprint(similar_bp,    url_prefix="/gemini/similar")
        app.register_blueprint(recommend_bp,  url_prefix="/gemini/recommend")

        logger.info("✓ Blueprints registered under /gemini")
    except ImportError as e:
        logger.warning("Could not import blueprints: %s", e)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Not found"}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}")
        return {"error": "Internal server error"}, 500
    
    # Root endpoint
    @app.route('/')
    def root():
        return {
            "service": "Gemini Product Search & Recommendation API",
            "version": "2.0",
            "base_path": "/gemini",
            "endpoints": {
                "health": "/gemini/health",
                "search": "/gemini/search",
                "similar": "/gemini/similar",
                "recommend": "/gemini/recommend",
                "events": "/gemini/events",
                "products": "/gemini/products",
                "index": "/gemini/index",
            }
        }, 200
    
    return app