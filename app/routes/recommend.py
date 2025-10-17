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

def _get_user_interacted_products(user_id: str, event_types: list = None):
    """Lấy danh sách product_id user đã tương tác, với trọng số theo loại event."""
    event_service = _event()
    events = event_service.get_user_events(user_id, event_types, limit=50)  # Giới hạn để tránh quá nhiều
    
    interacted = defaultdict(float)  # product_id -> weight (cao hơn nếu purchase/wishlist)
    weights = {"purchase": 3.0, "wishlist": 2.0, "cart": 1.5, "view": 1.0}
    
    for event in events:
        pid = event.get("product_id")
        etype = event.get("type")
        weight = weights.get(etype, 1.0)
        interacted[pid] += weight
    
    return dict(interacted)

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
    - top_k: Số lượng gợi ý (default: 5)
    - event_types: List loại event để filter (e.g., "purchase,wishlist", default: all)
    """
    try:
        top_k = int(request.args.get("top_k", 20))
        event_types_str = request.args.get("event_types", "")
        event_types = [t.strip() for t in event_types_str.split(",") if t.strip()] if event_types_str else None
        
        # 1. Lấy sản phẩm đã tương tác của user
        interacted = _get_user_interacted_products(user_id, event_types)
        if not interacted:
            # Fallback: Top popular products (dựa trên số events của product)
            return _get_popular_products(top_k)
        
        # 2. Tìm similar cho từng interacted product
        mongo = _mongo()
        col = mongo.db[current_app.config.get("COLLECTION_NAME", "products")]
        
        candidates = defaultdict(float)  # product_id -> aggregated_score
        exclude_ids = set(interacted.keys())  # Loại trừ đã tương tác
        
        for pid, weight in interacted.items():
            product = _get_product_by_any_id(col, pid)
            if not product:
                continue
            
            # Text để embedding (tương tự similar.py)
            name = product.get("name", "")
            desc = product.get("description", "")
            text = product.get("text_indexed", f"{name}. {desc}")
            
            # Tạo embedding và tìm neighbors
            emb = _vs().create_embedding(text, task_type="RETRIEVAL_DOCUMENT")
            if not emb:
                continue
            
            neighbors = _vs().find_neighbors(emb, k=top_k * 2)  # Lấy nhiều hơn để filter
            
            for nid, dist in neighbors:
                nid_str = str(nid)
                if nid_str in exclude_ids:
                    continue
                # Score: similarity (1 - dist, giả sử dist 0-1) * weight từ interacted
                sim_score = 1.0 - float(dist)  # Chuyển dist thành sim (càng giống càng cao)
                candidates[nid_str] += sim_score * weight
        
        # 3. Sort và lấy top
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        top_ids = [pid for pid, _ in sorted_candidates[:top_k]]
        
        # 4. Lấy chi tiết products
        results = []
        for pid in top_ids:
            product = _get_product_by_any_id(col, pid)
            if product:
                product["_id"] = str(product.get("_id", pid))
                results.append({
                    "product": product,
                    "recommend_score": candidates[pid]  # Score tổng
                })
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "total_recommendations": len(results),
            "recommendations": results
        }), 200
        
    except Exception as e:
        logger.exception("recommend_for_user failed")
        return jsonify({"error": str(e)}), 500

def _get_popular_products(top_k: int):
    """Fallback: Top products phổ biến dựa trên số events."""
    try:
        mongo = _mongo()
        events_col = mongo.db["user_events"]  # Giả sử collection events là "user_events"
        
        # Group by product_id và count
        pipeline = [
            {"$group": {"_id": "$product_id", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": top_k}
        ]
        popular = list(events_col.aggregate(pipeline))
        
        col = mongo.db[current_app.config.get("COLLECTION_NAME", "products")]
        results = []
        for doc in popular:
            pid = doc["_id"]
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