from flask import Blueprint, request, jsonify, current_app
import logging

logger = logging.getLogger(__name__)

events_bp = Blueprint('events_bp', __name__)

@events_bp.route('/track', methods=['POST'])
def track_event():
    """
    Track một event của user
    
    Body:
        {
            "user_id": "user123",
            "product_id": "68ccea231b70971f96da9f5a",
            "type": "view",  # view, cart, purchase, wishlist
            "metadata": {
                "price": 150000,
                "quantity": 1
            }
        }
    """
    try:
        data = request.get_json()
        
        user_id = data.get("user_id")
        product_id = data.get("product_id")
        event_type = data.get("type")
        metadata = data.get("metadata", {})
        
        if not all([user_id, product_id, event_type]):
            return jsonify({"error": "user_id, product_id, và type là bắt buộc"}), 400
        
        event_service = current_app.config['EVENT_SERVICE']
        event = event_service.track_event(user_id, product_id, event_type, metadata)
        
        return jsonify({
            "message": "Event tracked successfully",
            "event": event
        }), 201
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error tracking event: {e}")
        return jsonify({"error": str(e)}), 500


@events_bp.route('/batch', methods=['POST'])
def batch_track_events():
    """
    Track nhiều events cùng lúc
    
    Body:
        {
            "events": [
                {"user_id": "user123", "product_id": "prod1", "type": "view"},
                {"user_id": "user123", "product_id": "prod2", "type": "cart"}
            ]
        }
    """
    try:
        data = request.get_json()
        events = data.get("events", [])
        
        if not events or not isinstance(events, list):
            return jsonify({"error": "events phải là array"}), 400
        
        event_service = current_app.config['EVENT_SERVICE']
        result = event_service.batch_track_events(events)
        
        return jsonify({
            "message": "Batch tracking completed",
            "result": result
        }), 201
        
    except Exception as e:
        logger.error(f"Error batch tracking: {e}")
        return jsonify({"error": str(e)}), 500


@events_bp.route('/user/<user_id>', methods=['GET'])
def get_user_events(user_id):
    """
    Lấy lịch sử events của user
    
    Query params:
        - type: Filter theo loại event (optional)
        - limit: Số events (default: 100)
    """
    try:
        event_type = request.args.get("type")
        limit = request.args.get("limit", default=100, type=int)
        
        event_service = current_app.config['EVENT_SERVICE']
        events = event_service.get_user_events(user_id, event_type, limit)
        
        return jsonify({
            "user_id": user_id,
            "total_events": len(events),
            "events": events
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting user events: {e}")
        return jsonify({"error": str(e)}), 500


@events_bp.route('/user/<user_id>/stats', methods=['GET'])
def get_user_stats(user_id):
    """Thống kê events của user"""
    try:
        event_service = current_app.config['EVENT_SERVICE']
        stats = event_service.get_user_stats(user_id)
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return jsonify({"error": str(e)}), 500


@events_bp.route('/product/<product_id>', methods=['GET'])
def get_product_events(product_id):
    """Lấy events của một product"""
    try:
        event_type = request.args.get("type")
        limit = request.args.get("limit", default=100, type=int)
        
        event_service = current_app.config['EVENT_SERVICE']
        events = event_service.get_product_events(product_id, event_type, limit)
        
        return jsonify({
            "product_id": product_id,
            "total_events": len(events),
            "events": events
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting product events: {e}")
        return jsonify({"error": str(e)}), 500