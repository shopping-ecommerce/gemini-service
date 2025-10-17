# app/routes/search.py
from flask import Blueprint, current_app, request, jsonify
from bson import ObjectId

search_bp = Blueprint("search_bp", __name__)

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

def _get_product_by_any_id(col, nid: str):
    """Thử map id trả về từ VS vào Mongo:
       - Nếu nid là ObjectId hợp lệ → tìm theo _id(ObjectId)
       - Nếu không → thử _id string
       - Cuối cùng → thử field 'id' (nếu có)"""
    # 1) _id = ObjectId
    if ObjectId.is_valid(nid):
        p = col.find_one({"_id": ObjectId(nid)})
        if p:
            return p
    # 2) _id = string
    p = col.find_one({"_id": nid})
    if p:
        return p
    # 3) field 'id'
    return col.find_one({"id": nid})

@search_bp.route("/search", methods=["POST"])
def semantic_search():
    """
    Body:
    {
      "query": "...",
      "top_k": 5,
      "filter": { "category":"...", "price_min":..., "price_max":... }
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        top_k = int(data.get("top_k", 5))
        filters = data.get("filter") or {}

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # 1) Tạo embedding cho truy vấn
        q_emb = _vs().create_embedding(query, task_type="RETRIEVAL_QUERY")
        if not q_emb:
            return jsonify({"error": "Failed to create query embedding"}), 500

        # 2) Tìm hàng xóm trong Vertex Vector Search
        #    Trả về list[(datapoint_id, distance)]
        neighbors = _vs().find_neighbors(q_emb, k=top_k)

        # 3) Lấy chi tiết sản phẩm từ Mongo + áp filter (nếu có)
        mongo = _mongo()
        col = mongo.db[current_app.config.get("COLLECTION_NAME", "products")]
        emb_col = mongo.db[current_app.config.get("EMBEDDINGS_COLLECTION", "product_embeddings")]

        results = []
        for nid, dist in neighbors:
            product = _get_product_by_any_id(col, str(nid))
            if not product:
                continue

            # Apply filters nếu bạn muốn bật lại
            # if 'category' in filters and product.get('category') != filters['category']:
            #     continue
            # price = product.get('price') or 0
            # if 'price_min' in filters and price < float(filters['price_min']):
            #     continue
            # if 'price_max' in filters and price > float(filters['price_max']):
            #     continue

            # Chuẩn hoá _id về string
            if "_id" in product:
                product["_id"] = str(product["_id"])
                emb_doc = emb_col.find_one({"product_id": str(product["_id"])}, {"text": 1})
                raw_text = (emb_doc or {}).get("text") or f"{product.get('name','')}. {product.get('description','')}"
                snippet = (raw_text[:400] + "…") if len(raw_text) > 400 else raw_text
                results.append({
                "product": product,
                # Lưu ý: distance của VS càng NHỎ càng giống.
                # Nếu muốn similarity 0..1: sim = (-distance + 1) / 2 (tuỳ cấu hình index)
                "similarity_score": float(dist),
                "matched_text": snippet
            })

        return jsonify({
            "success": True,
            "query": query,
            "total_results": len(results),
            "results": results
        }), 200

    except Exception as e:
        current_app.logger.exception("semantic_search failed")
        return jsonify({"error": str(e)}), 500
