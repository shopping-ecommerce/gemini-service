# app/routes/recommend.py
from flask import Blueprint, request, jsonify, current_app
from bson import ObjectId
import logging
from collections import defaultdict

recommend_bp = Blueprint("recommend_bp", __name__)
logger = logging.getLogger(__name__)

def _vs():
    svc = current_app.config.get("VERTEX_AI_SERVICE")
    if not svc:
        raise RuntimeError("VERTEX_AI_SERVICE chưa được khởi tạo")
    return svc

def _mongo():
    svc = current_app.config.get("MONGODB_SERVICE")
    if not svc:
        raise RuntimeError("MONGODB_SERVICE chưa được khởi tạo")
    return svc

def _event():
    svc = current_app.config.get("EVENT_SERVICE")
    if not svc:
        raise RuntimeError("EVENT_SERVICE chưa được khởi tạo")
    return svc

def _get_product_by_any_id(col, pid: str):
    """Tìm product theo nhiều khả năng id (reuse từ similar.py)."""
    if ObjectId.is_valid(pid):
        p = col.find_one({"_id": ObjectId(pid)})
        if p:
            return p
    p = col.find_one({"_id": pid})
    if p:
        return p
    return col.find_one({"id": pid})

def _get_user_interacted_products(user_id: str, event_types: list = None, limit: int = 50):
    """Lấy danh sách product_id user đã tương tác, với trọng số theo loại event."""
    from datetime import datetime
    
    mongo = _mongo()
    events_col = mongo.db["events"]  # Collection name từ screenshot
    
    # Build query - FIX: Dùng userId thay vì user_id
    query = {"userId": user_id}
    if event_types:
        query["type"] = {"$in": event_types}
    
    # Sort theo ts (timestamp field trong MongoDB)
    events = list(events_col.find(query).sort("ts", -1).limit(limit))
    
    logger.info(f"[DEBUG] User {user_id} has {len(events)} events from MongoDB")
    
    if not events:
        logger.warning(f"[DEBUG] No events found for userId={user_id}")
        return {}
    
    interacted = defaultdict(lambda: {"weight": 0.0, "latest_time": None})
    weights = {"purchase": 3.0, "wishlist": 2.0, "cart": 1.5, "view": 1.0}
    
    for event in events:
        # FIX: Dùng productId thay vì product_id
        pid = event.get("productId")
        if not pid:
            logger.warning(f"[DEBUG] Event missing productId: {event.get('_id')}")
            continue
        
        etype = event.get("type")
        # FIX: Dùng ts thay vì timestamp
        timestamp = event.get("ts") or event.get("timestamp") or event.get("created_at")
        
        weight = weights.get(etype, 1.0)
        
        # Time decay
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    event_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    # ts có thể là datetime object từ MongoDB
                    event_time = timestamp
                
                # Ensure timezone aware
                if event_time.tzinfo is None:
                    from datetime import timezone
                    event_time = event_time.replace(tzinfo=timezone.utc)
                
                now = datetime.now(event_time.tzinfo)
                days_old = (now - event_time).days
                decay = max(0.5, 1.0 - (days_old / 365))
                weight *= decay
            except Exception as e:
                logger.warning(f"[DEBUG] Failed to parse timestamp {timestamp}: {e}")
        
        interacted[pid]["weight"] += weight
        if not interacted[pid]["latest_time"] or (timestamp and timestamp > interacted[pid]["latest_time"]):
            interacted[pid]["latest_time"] = timestamp
    
    logger.info(f"[DEBUG] Found {len(interacted)} unique products for user {user_id}")
    if interacted:
        top_3 = list(interacted.items())[:3]
        logger.info(f"[DEBUG] Top 3 products: {top_3}")
    
    result = {k: v["weight"] for k, v in interacted.items()}
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))



def _get_product(col, pid: str):
    # Tìm theo nhiều kiểu id: ObjectId(_id) -> _id string -> field "id"
    if ObjectId.is_valid(pid):
        p = col.find_one({"_id": ObjectId(pid)})
        if p: return p
    p = col.find_one({"_id": pid})
    if p: return p
    return col.find_one({"id": pid})

@recommend_bp.route("/user/<user_id>", methods=["GET"])
def recommend_for_user(user_id):
    """
    Gợi ý sản phẩm dựa trên lịch sử user.
    
    Query params:
    - top_k: Số lượng gợi ý (default: 20)
    - event_types: List loại event để filter (e.g., "purchase,wishlist", default: all)
    - diversity: Mức độ đa dạng 0.0-1.0 (default: 0.3)
    """
    try:
        top_k = int(request.args.get("top_k", 20))
        diversity = float(request.args.get("diversity", 0.3))
        event_types_str = request.args.get("event_types", "")
        event_types = [t.strip() for t in event_types_str.split(",") if t.strip()] if event_types_str else None
        
        # 1. Lấy sản phẩm đã tương tác của user
        interacted = _get_user_interacted_products(user_id, event_types, limit=100)
        
        if not interacted:
            logger.info(f"User {user_id} has no interaction history, returning popular products")
            return _get_popular_products(top_k)
        
        logger.info(f"User {user_id} has {len(interacted)} interacted products")
        
        # 2. Tạo user preference embedding từ top interacted products
        mongo = _mongo()
        col = mongo.db[current_app.config.get("COLLECTION_NAME", "products")]
        
        # Lấy top 10 products quan trọng nhất để tạo profile
        top_interacted = list(interacted.items())[:10]
        user_texts = []
        
        for pid, weight in top_interacted:
            product = _get_product_by_any_id(col, pid)
            if not product:
                logger.warning(f"[DEBUG] Product {pid} not found in products collection")
                continue
            text = product.get("text_indexed") or f"{product.get('name', '')}. {product.get('description', '')}"
            user_texts.append(text)
        
        if not user_texts:
            logger.warning("No valid products found from user history")
            return _get_popular_products(top_k)
        
        logger.info(f"Creating user profile from {len(user_texts)} products")
        
        # Tạo user profile embedding (trung bình có trọng số)
        user_profile_text = " | ".join(user_texts[:5])  # Ghép top 5 products
        user_emb = _vs().create_embedding(user_profile_text, task_type="RETRIEVAL_QUERY")
        
        if not user_emb:
            logger.warning("Failed to create user embedding, fallback to individual products")
            return _recommend_by_individual_products(user_id, interacted, col, top_k)
        
        # 3. Tìm candidates từ vector search
        exclude_ids = set(interacted.keys())
        neighbors = _vs().find_neighbors(user_emb, k=top_k * 3)  # Lấy nhiều để filter
        
        candidates = {}
        seen_categories = defaultdict(int)  # Track category diversity
        
        for nid, dist in neighbors:
            nid_str = str(nid)
            if nid_str in exclude_ids:
                continue
            
            product = _get_product_by_any_id(col, nid_str)
            if not product:
                continue
            
            # Tính similarity score (1 - distance)
            sim_score = max(0, 1.0 - float(dist))
            
            # Diversity penalty: giảm score nếu category đã có nhiều
            category = product.get("category") or product.get("categoryId")
            if category and diversity > 0:
                category_count = seen_categories.get(category, 0)
                diversity_penalty = diversity * (category_count * 0.1)  # Mỗi item cùng category giảm 10%
                sim_score *= (1.0 - min(0.5, diversity_penalty))
            
            candidates[nid_str] = {
                "score": sim_score,
                "product": product,
                "category": category
            }
            
            if category:
                seen_categories[category] += 1
            
            if len(candidates) >= top_k * 2:
                break
        
        # 4. Sort và lấy top
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]
        
        results = []
        for pid, data in sorted_candidates:
            product = data["product"]
            product["_id"] = str(product.get("_id", pid))
            results.append({
                "product": product,
                "recommend_score": round(data["score"], 4)
            })
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "strategy": "personalized_content_based",
            "total_interactions": len(interacted),
            "total_recommendations": len(results),
            "recommendations": results
        }), 200
        
    except Exception as e:
        logger.exception("recommend_for_user failed")
        return jsonify({"error": str(e)}), 500

def _recommend_by_individual_products(user_id, interacted, col, top_k):
    """Fallback: Tìm similar từng product rồi aggregate (logic cũ nhưng cải thiện)"""
    candidates = defaultdict(lambda: {"score": 0.0, "count": 0})
    exclude_ids = set(interacted.keys())
    
    # Chỉ lấy top 5 interacted products quan trọng nhất
    top_interacted = list(interacted.items())[:5]
    
    for pid, weight in top_interacted:
        product = _get_product_by_any_id(col, pid)
        if not product:
            continue
        
        text = product.get("text_indexed") or f"{product.get('name', '')}. {product.get('description', '')}"
        emb = _vs().create_embedding(text, task_type="RETRIEVAL_DOCUMENT")
        if not emb:
            continue
        
        neighbors = _vs().find_neighbors(emb, k=top_k * 2)
        
        for nid, dist in neighbors:
            nid_str = str(nid)
            if nid_str in exclude_ids:
                continue
            
            sim_score = max(0, 1.0 - float(dist))
            # Normalize by count để tránh bias
            candidates[nid_str]["score"] += sim_score * weight
            candidates[nid_str]["count"] += 1
    
    # Normalize scores
    for pid in candidates:
        if candidates[pid]["count"] > 0:
            candidates[pid]["score"] /= candidates[pid]["count"]
    
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]
    
    results = []
    for pid, data in sorted_candidates:
        product = _get_product_by_any_id(col, pid)
        if product:
            product["_id"] = str(product.get("_id", pid))
            results.append({
                "product": product,
                "recommend_score": round(data["score"], 4)
            })
    
    return jsonify({
        "success": True,
        "user_id": user_id,
        "strategy": "aggregated_content_based",
        "total_recommendations": len(results),
        "recommendations": results
    }), 200


def _get_popular_products(top_k: int):
    """Fallback: Top products phổ biến dựa trên số events."""
    try:
        mongo = _mongo()
        events_col = mongo.db["events"]  # FIX: Collection name
        
        # Group by productId (not product_id)
        pipeline = [
            {"$group": {"_id": "$productId", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": top_k}
        ]
        popular = list(events_col.aggregate(pipeline))
        
        col = mongo.db[current_app.config.get("COLLECTION_NAME", "products")]
        results = []
        for doc in popular:
            pid = doc["_id"]
            if not pid:
                continue
            product = _get_product_by_any_id(col, pid)
            if product:
                product["_id"] = str(product.get("_id", pid))
                results.append({
                    "product": product,
                    "popularity_score": doc["count"]
                })
        
        return jsonify({
            "success": True,
            "user_id": "fallback",
            "message": "No history, showing popular products",
            "total_recommendations": len(results),
            "recommendations": results
        }), 200
        
    except Exception as e:
        logger.exception("_get_popular_products failed")
        return jsonify({"error": "Failed to get popular products"}), 500
# =======================
# 2) Recommend theo PRODUCT (content-based)
# =======================
@recommend_bp.route("/for-product/<product_id>", methods=["GET"])
def recommend_for_product(product_id):
    """
    Query:
      - top_k (default 10)
      - include_self (default false)
    """
    try:
        top_k = int(request.args.get("top_k", 10))
        include_self = request.args.get("include_self", "false").lower() == "true"

        mongo = _mongo()
        prod_col = mongo.db[current_app.config.get("COLLECTION_NAME", "products")]

        prod = _get_product(prod_col, str(product_id))
        if not prod:
            return jsonify({"error": "Product not found"}), 404

        # Ưu tiên text_indexed nếu có
        text = prod.get("text_indexed") or f"{prod.get('name','')}. {prod.get('description','')}"
        emb = _vs().create_embedding(text, task_type="RETRIEVAL_DOCUMENT")
        if not emb:
            return jsonify({"error": "Failed to create embedding"}), 500

        neighbors = _vs().find_neighbors(emb, k=top_k + 1)
        self_id = str(prod.get("_id") or prod.get("id") or product_id)

        results = []
        for nid, dist in neighbors:
            nid = str(nid)
            if not include_self and nid == self_id:
                continue
            p = _get_product(prod_col, nid)
            if not p:
                continue
            p["_id"] = str(p.get("_id", nid))
            results.append({
                "product": p,
                "similarity_score": dist
            })
            if len(results) >= top_k:
                break

        return jsonify({
            "success": True,
            "strategy": "content_based",
            "product_id": self_id,
            "results": results
        }), 200

    except Exception as e:
        current_app.logger.exception("recommend_for_product failed")
        return jsonify({"error": str(e)}), 500