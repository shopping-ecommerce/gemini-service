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
