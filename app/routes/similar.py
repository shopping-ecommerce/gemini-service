# app/routes/similar.py
from flask import Blueprint, jsonify, request, current_app
from bson import ObjectId

similar_bp = Blueprint("similar_bp", __name__)

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

def _get_product_by_any_id(col, pid: str):
    """Tìm product theo nhiều khả năng id."""
    if ObjectId.is_valid(pid):
        p = col.find_one({"_id": ObjectId(pid)})
        if p:
            return p
    p = col.find_one({"_id": pid})
    if p:
        return p
    return col.find_one({"id": pid})

@similar_bp.route("/api/search/similar/<product_id>", methods=["GET"])
def find_similar_products(product_id):
    try:
        top_k = int(request.args.get("top_k", 5))

        # Mongo collections
        mongo = _mongo()
        col = mongo.db[current_app.config.get("COLLECTION_NAME", "products")]

        # Lấy sản phẩm gốc
        product = _get_product_by_any_id(col, str(product_id))
        if not product:
            return jsonify({"error": "Product not found"}), 404

        # Text để embedding
        name = product.get("name", "")
        desc = product.get("description", "")
        text = product.get("text_indexed", f"{name}. {desc}")

        # Tạo embedding và tìm hàng xóm
        emb = _vs().create_embedding(text, task_type="RETRIEVAL_DOCUMENT")
        if not emb:
            return jsonify({"error": "Failed to create embedding"}), 500

        neighbors = _vs().find_neighbors(emb, k=top_k + 1)  # [(id, distance)]

        # Build kết quả, bỏ chính nó
        self_id = str(product.get("_id") or product.get("id") or product_id)
        results = []
        for nid, dist in neighbors:
            nid = str(nid)
            if nid == self_id:
                continue
            sp = _get_product_by_any_id(col, nid)
            if not sp:
                continue
            sp["_id"] = str(sp.get("_id", nid))
            results.append({
                "product": sp,
                # distance nhỏ => giống hơn; nếu muốn similarity 0..1 thì tự chuyển đổi
                "similarity_score": float(dist)
            })

        return jsonify({
            "success": True,
            "product_id": self_id,
            "similar_products": results[:top_k]
        }), 200

    except Exception as e:
        current_app.logger.exception("find_similar_products failed")
        return jsonify({"error": str(e)}), 500
