# app/routes/products.py
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
from bson.objectid import ObjectId

products_bp = Blueprint("products_bp", __name__)

def _vs():
    svc = current_app.config.get("VERTEX_AI_SERVICE")
    if not svc:
        raise RuntimeError("VERTEX_AI_SERVICE chưa được khởi tạo trong app.config")
    return svc

def _mongo():
    svc = current_app.config.get("MONGODB_SERVICE")
    if not svc:
        raise RuntimeError("MONGODB_SERVICE chưa được khởi tạo trong app.config")
    return svc

@products_bp.route("/", methods=["POST"])
def add_product():
    try:
        data = request.get_json(silent=True) or {}
        if not data.get("name") or not data.get("description"):
            return jsonify({"error": "Name and description are required"}), 400

        text = f"{data['name']}. {data['description']}"
        if data.get("category"):
            text += f". Danh mục: {data['category']}"

        # Tạo embedding
        embedding = _vs().create_embedding(text, task_type="RETRIEVAL_DOCUMENT")
        if not embedding:
            return jsonify({"error": "Failed to create embedding"}), 500

        mongo = _mongo()
        products_col = mongo.db["products"]
        embeddings_col = mongo.db["product_embeddings"]

        product = {
            "name": data["name"],
            "description": data["description"],
            "category": data.get("category", ""),
            "price": data.get("price", 0),
            "metadata": data.get("metadata", {}),
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        result = products_col.insert_one(product)
        product_id = str(result.inserted_id)

        embeddings_col.insert_one({
            "product_id": product_id,
            "embedding": embedding,
            "text": text,
            "created_at": datetime.now()
        })

        # Upsert vào Vector Search (không làm fail toàn request nếu lỗi)
        try:
            _vs().upsert_vector(product_id, embedding)
        except Exception as ve:
            current_app.logger.warning(f"Upsert to Vector Search failed: {ve}")

        return jsonify({"success": True, "product_id": product_id}), 201

    except Exception as e:
        current_app.logger.exception("add_product failed")
        return jsonify({"error": str(e)}), 500


@products_bp.route("/batch", methods=["POST"])
def add_products_batch():
    try:
        payload = request.get_json(silent=True) or {}
        items = payload.get("products", [])
        if not items:
            return jsonify({"error": "No products provided"}), 400

        mongo = _mongo()
        products_col = mongo.db["products"]
        embeddings_col = mongo.db["product_embeddings"]

        added, errors = 0, []
        to_upsert = []

        for idx, p in enumerate(items):
            try:
                if not p.get("name") or not p.get("description"):
                    errors.append(f"Product {idx}: name/description missing")
                    continue

                text = f"{p['name']}. {p['description']}"
                if p.get("category"):
                    text += f". Danh mục: {p['category']}"

                emb = _vs().create_embedding(text, task_type="RETRIEVAL_DOCUMENT")
                if not emb:
                    errors.append(f"Product {idx}: embedding failed")
                    continue

                doc = {
                    "name": p["name"],
                    "description": p["description"],
                    "category": p.get("category", ""),
                    "price": p.get("price", 0),
                    "metadata": p.get("metadata", {}),
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                }
                res = products_col.insert_one(doc)
                pid = str(res.inserted_id)

                embeddings_col.insert_one({
                    "product_id": pid,
                    "embedding": emb,
                    "text": text,
                    "created_at": datetime.now()
                })
                added += 1
                to_upsert.append((pid, emb))

            except Exception as ex:
                errors.append(f"Product {idx}: {ex}")

        if to_upsert:
            try:
                _vs().upsert_vectors(to_upsert)
            except Exception as ve:
                errors.append(f"Batch upsert to VS failed: {ve}")

        return jsonify({
            "success": True,
            "added_count": added,
            "total": len(items),
            "errors": errors
        }), 201

    except Exception as e:
        current_app.logger.exception("add_products_batch failed")
        return jsonify({"error": str(e)}), 500


@products_bp.route("/", methods=["GET"])
def get_products():
    try:
        mongo = _mongo()
        products_col = mongo.db["products"]

        limit = int(request.args.get("limit", 100))
        skip = int(request.args.get("skip", 0))

        docs = list(products_col.find().skip(skip).limit(limit))
        for d in docs:
            d["_id"] = str(d["_id"])
        total = products_col.count_documents({})
        return jsonify({"success": True, "total": total, "products": docs}), 200
    except Exception as e:
        current_app.logger.exception("get_products failed")
        return jsonify({"error": str(e)}), 500


@products_bp.route("/<product_id>", methods=["DELETE"])
def delete_product(product_id):
    try:
        mongo = _mongo()
        products_col = mongo.db["products"]
        embeddings_col = mongo.db["product_embeddings"]

        try:
            oid = ObjectId(product_id)
        except Exception:
            return jsonify({"error": "Invalid product_id"}), 400

        result = products_col.delete_one({"_id": oid})
        if result.deleted_count == 0:
            return jsonify({"error": "Product not found"}), 404

        embeddings_col.delete_many({"product_id": product_id})

        # Remove vector trong VS (không fail toàn request nếu lỗi)
        try:
            _vs().remove_vectors([product_id])
        except Exception as ve:
            current_app.logger.warning(f"Remove from Vector Search failed: {ve}")

        return jsonify({"success": True, "message": "Product deleted successfully"}), 200

    except Exception as e:
        current_app.logger.exception("delete_product failed")
        return jsonify({"error": str(e)}), 500
