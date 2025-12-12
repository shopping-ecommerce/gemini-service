# app/routes/index_ops.py
from flask import Blueprint, jsonify, current_app
from datetime import datetime
import logging, time

index_bp = Blueprint("index_bp", __name__)
logger = logging.getLogger(__name__)

def _vs():
    svc = current_app.config.get("VERTEX_AI_SERVICE")
    if not svc:
        raise RuntimeError("VERTEX_AI_SERVICE chưa được khởi tạo")
    return svc

def _chunks(items, n):
    for i in range(0, len(items), n):
        yield items[i:i+n]

@index_bp.route("/rebuild-index", methods=["POST"])
def rebuild_index():
    try:
        mongo = current_app.config["MONGODB_SERVICE"]
        products_col = mongo.db["products"]
        embeddings_col = mongo.db["product_embeddings"]

        # (Nếu muốn clean full) Lấy old_ids TRƯỚC khi xoá Mongo
        try:
            old_ids = [doc.get("product_id") for doc in embeddings_col.find({}, {"product_id": 1})]
            old_ids = [pid for pid in old_ids if pid]
        except Exception as ve:
            logger.warning("List old ids failed: %s", ve)
            old_ids = []

        # Xoá embeddings cũ ở Mongo
        embeddings_col.delete_many({})

        # Xoá vector cũ ở Vertex VS (nếu cần)
        if old_ids:
            try:
                _vs().remove_vectors(old_ids)
                logger.info("Removed %d old vectors from Vertex VS", len(old_ids))
            except Exception as ve:
                logger.warning("Could not remove old vectors from VS: %s", ve)

        # Chuẩn bị dữ liệu cần embed
        to_embed = []
        for p in products_col.find({}, {"_id":1, "name":1, "description":1}):
            pid = str(p["_id"])
            name = p.get("name", "")
            desc = p.get("description", "")
            text = f"{name}. {desc}"
            # (tuỳ) cắt ngắn để giảm chi phí
            if len(text) > 4000:
                text = text[:4000]
            to_embed.append((pid, text))

        batch_size = 100   # tuỳ quota
        total_upsert = 0

        for batch in _chunks(to_embed, batch_size):
            ids = [pid for pid, _ in batch]
            texts = [txt for _, txt in batch]

            # Embedding batch (đã có retry/backoff bên trong service)
            try:
                embs = _vs().create_embeddings_batch(texts, task_type="RETRIEVAL_DOCUMENT")
            except Exception as ge:
                logger.warning("Batch embedding failed (%d items): %s", len(texts), ge)
                # nếu fail cả batch, tiếp tục batch sau
                time.sleep(1.2)
                continue

            now = datetime.now()
            docs = []
            pairs_for_upsert = []
            for pid, emb, txt in zip(ids, embs, texts):
                if not emb:
                    continue
                docs.append({
                    "product_id": pid,
                    "embedding": emb,
                    "text": txt,
                    "created_at": now,
                })
                pairs_for_upsert.append((pid, emb))

            if docs:
                embeddings_col.insert_many(docs, ordered=False)

            if pairs_for_upsert:
                try:
                    _vs().upsert_vectors(pairs_for_upsert)
                    total_upsert += len(pairs_for_upsert)
                    logger.info("Upserted %d/%d this batch, total=%d",
                                len(pairs_for_upsert), len(batch), total_upsert)
                except Exception as ve:
                    logger.error("Failed to upsert to Vertex VS (batch): %s", ve)

            # rate-limit nhẹ giữa các batch để tránh 429
            time.sleep(0.8)

        return jsonify({"success": True, "rebuilt_count": total_upsert}), 200

    except Exception as e:
        logger.error("Rebuild index failed: %s", e)
        return jsonify({"error": str(e)}), 500


@index_bp.route("/index-single-product", methods=["POST"])
def index_single_product():
    """
    Index text embedding cho MỘT product cụ thể
    
    Request body:
    {
        "product_id": "67123abc...",
        "force_reindex": false  // optional: xóa và tạo lại nếu đã tồn tại
    }
    """
    try:
        from flask import request
        from bson import ObjectId
        
        data = request.get_json()
        if not data or "product_id" not in data:
            return jsonify({"error": "product_id is required"}), 400
        
        product_id = data["product_id"]
        force_reindex = data.get("force_reindex", False)
        
        mongo = current_app.config["MONGODB_SERVICE"]
        products_col = mongo.db["products"]
        embeddings_col = mongo.db["product_embeddings"]
        
        # Validate product exists
        try:
            oid = ObjectId(product_id) if ObjectId.is_valid(product_id) else product_id
        except Exception:
            oid = product_id
        
        product = products_col.find_one({"_id": oid})
        if not product:
            return jsonify({"error": f"Product {product_id} not found"}), 404
        
        pid = str(product["_id"])
        
        # Kiểm tra đã tồn tại chưa
        existing = embeddings_col.find_one({"product_id": pid})
        if existing and not force_reindex:
            return jsonify({
                "success": True,
                "product_id": pid,
                "message": "Product already indexed (use force_reindex=true to recreate)"
            }), 200
        
        # Chuẩn bị text
        name = product.get("name", "")
        desc = product.get("description", "")
        text = f"{name}. {desc}"
        
        if len(text) > 4000:
            text = text[:4000]
        
        if not text.strip():
            return jsonify({
                "success": False,
                "product_id": pid,
                "error": "Product has no text content to index"
            }), 400
        
        # Tạo embedding
        try:
            emb = _vs().create_embedding(text, task_type="RETRIEVAL_DOCUMENT")
        except Exception as e:
            logger.error(f"Failed to create embedding for product {pid}: {e}")
            return jsonify({"error": f"Failed to create embedding: {str(e)}"}), 500
        
        if not emb:
            return jsonify({"error": "Failed to create embedding"}), 500
        
        # Upsert vào MongoDB
        now = datetime.now()
        doc = {
            "product_id": pid,
            "embedding": emb,
            "text": text,
            "created_at": now,
            "updated_at": now,
        }
        
        embeddings_col.replace_one(
            {"product_id": pid},
            doc,
            upsert=True
        )
        
        # Upsert vào Vertex AI
        try:
            _vs().upsert_vector(pid, emb)
            logger.info(f"Indexed product {pid}")
        except Exception as ve:
            logger.error(f"Failed to upsert to Vertex VS: {ve}")
            return jsonify({"error": f"Failed to upsert to Vertex: {str(ve)}"}), 500
        
        return jsonify({
            "success": True,
            "product_id": pid,
            "message": "Successfully indexed product"
        }), 200
        
    except Exception as e:
        logger.error(f"Index single product failed: {e}")
        return jsonify({"error": str(e)}), 500


@index_bp.route("/remove-single-product", methods=["POST"])
def remove_single_product():
    """
    Xóa text embedding của MỘT product
    
    Request body:
    {
        "product_id": "67123abc..."
    }
    """
    try:
        from flask import request
        
        data = request.get_json()
        if not data or "product_id" not in data:
            return jsonify({"error": "product_id is required"}), 400
        
        product_id = data["product_id"]
        
        mongo = current_app.config["MONGODB_SERVICE"]
        embeddings_col = mongo.db["product_embeddings"]
        
        # Kiểm tra tồn tại
        existing = embeddings_col.find_one({"product_id": product_id})
        if not existing:
            return jsonify({
                "success": True,
                "product_id": product_id,
                "message": "Embedding not found (already removed)"
            }), 200
        
        # Xóa từ MongoDB
        embeddings_col.delete_one({"product_id": product_id})
        
        # Xóa từ Vertex AI
        try:
            _vs().remove_vectors([product_id])
            logger.info(f"Removed product vector {product_id}")
        except Exception as ve:
            logger.warning(f"Could not remove vector from Vertex: {ve}")
        
        return jsonify({
            "success": True,
            "product_id": product_id,
            "message": "Successfully removed product embedding"
        }), 200
        
    except Exception as e:
        logger.error(f"Remove single product failed: {e}")
        return jsonify({"error": str(e)}), 500


@index_bp.route("/upsert-single-product", methods=["POST"])
def upsert_single_product():
    """
    Thêm hoặc cập nhật text embedding cho MỘT product
    (Alias của index-single-product với force_reindex=True)
    
    Request body:
    {
        "product_id": "67123abc..."
    }
    """
    try:
        from flask import request
        
        data = request.get_json()
        if not data or "product_id" not in data:
            return jsonify({"error": "product_id is required"}), 400
        
        # Force reindex
        data["force_reindex"] = True
        
        # Reuse index_single_product logic
        from bson import ObjectId
        
        product_id = data["product_id"]
        
        mongo = current_app.config["MONGODB_SERVICE"]
        products_col = mongo.db["products"]
        embeddings_col = mongo.db["product_embeddings"]
        
        # Validate product exists
        try:
            oid = ObjectId(product_id) if ObjectId.is_valid(product_id) else product_id
        except Exception:
            oid = product_id
        
        product = products_col.find_one({"_id": oid})
        if not product:
            return jsonify({"error": f"Product {product_id} not found"}), 404
        
        pid = str(product["_id"])
        
        # Chuẩn bị text
        name = product.get("name", "")
        desc = product.get("description", "")
        text = f"{name}. {desc}"
        
        if len(text) > 4000:
            text = text[:4000]
        
        if not text.strip():
            return jsonify({
                "success": False,
                "product_id": pid,
                "error": "Product has no text content to index"
            }), 400
        
        # Tạo embedding
        try:
            emb = _vs().create_embedding(text, task_type="RETRIEVAL_DOCUMENT")
        except Exception as e:
            logger.error(f"Failed to create embedding for product {pid}: {e}")
            return jsonify({"error": f"Failed to create embedding: {str(e)}"}), 500
        
        if not emb:
            return jsonify({"error": "Failed to create embedding"}), 500
        
        # Upsert vào MongoDB
        now = datetime.now()
        doc = {
            "product_id": pid,
            "embedding": emb,
            "text": text,
            "updated_at": now,
        }
        
        # Nếu đã tồn tại, giữ nguyên created_at
        existing = embeddings_col.find_one({"product_id": pid})
        if existing:
            doc["created_at"] = existing.get("created_at", now)
        else:
            doc["created_at"] = now
        
        embeddings_col.replace_one(
            {"product_id": pid},
            doc,
            upsert=True
        )
        
        # Upsert vào Vertex AI
        try:
            _vs().upsert_vector(pid, emb)
            logger.info(f"Upserted product {pid}")
        except Exception as ve:
            logger.error(f"Failed to upsert to Vertex VS: {ve}")
            return jsonify({"error": f"Failed to upsert to Vertex: {str(ve)}"}), 500
        
        return jsonify({
            "success": True,
            "product_id": pid,
            "message": "Successfully upserted product embedding"
        }), 200
        
    except Exception as e:
        logger.error(f"Upsert single product failed: {e}")
        return jsonify({"error": str(e)}), 500