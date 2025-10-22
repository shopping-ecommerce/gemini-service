from flask import Blueprint, current_app, jsonify
from datetime import datetime
import logging
import time
import requests

multi_image_index_bp = Blueprint("multi_image_index_bp", __name__)
logger = logging.getLogger(__name__)

def _vs():
    svc = current_app.config.get("VERTEX_AI_SERVICE")
    if not svc:
        raise RuntimeError("VERTEX_AI_SERVICE chưa được khởi tạo")
    return svc

def _chunks(items, n):
    for i in range(0, len(items), n):
        yield items[i:i+n]


@multi_image_index_bp.route("/rebuild-image-index-multi", methods=["POST"])
def rebuild_image_index_multi():
    """
    Index TẤT CẢ hình ảnh của mỗi product
    
    Mỗi ảnh sẽ có datapoint_id riêng: "{product_id}_{position}"
    Ví dụ: 
    - "689af8b9_1" → image position 1
    - "689af8b9_2" → image position 2
    """
    try:
        mongo = current_app.config["MONGODB_SERVICE"]
        products_col = mongo.db["products"]
        image_embeddings_col = mongo.db["product_image_embeddings"]

        # Xóa embeddings cũ
        try:
            old_ids = [doc.get("datapoint_id") for doc in image_embeddings_col.find({}, {"datapoint_id": 1})]
            old_ids = [did for did in old_ids if did]
        except Exception as e:
            logger.warning("List old image ids failed: %s", e)
            old_ids = []

        image_embeddings_col.delete_many({})

        # Xóa vector cũ từ Vertex
        if old_ids:
            try:
                _vs().remove_image_vectors(old_ids)
                logger.info("Removed %d old image vectors from Vertex", len(old_ids))
            except Exception as ve:
                logger.warning("Could not remove old image vectors: %s", ve)

        # Lấy tất cả images từ tất cả products
        to_embed = []
        for p in products_col.find({}, {"_id": 1, "name": 1, "images": 1}):
            pid = str(p["_id"])
            images = p.get("images", [])
            
            if not images or len(images) == 0:
                continue
            
            # Xử lý TẤT CẢ images, không chỉ ảnh đầu tiên
            for img in images:
                if isinstance(img, dict):
                    url = img.get("url")
                    position = img.get("position", 999)
                elif isinstance(img, str):
                    url = img
                    position = 999
                else:
                    continue
                
                if not url or not isinstance(url, str):
                    continue
                
                if not url.startswith(('http://', 'https://')):
                    continue
                
                # Tạo unique datapoint_id cho mỗi ảnh
                datapoint_id = f"{pid}_{position}"
                
                to_embed.append({
                    "datapoint_id": datapoint_id,
                    "product_id": pid,
                    "image_url": url,
                    "position": position,
                    "product_name": p.get("name", "")
                })

        logger.info(f"Found {len(to_embed)} images from products")
        
        if to_embed:
            logger.info(f"Sample images: {to_embed[:3]}")

        # Giảm batch_size để tránh quota
        batch_size = 3  # Giảm từ 10 xuống 3
        total_upsert = 0
        failed_count = 0

        for batch in _chunks(to_embed, batch_size):
            batch_embeddings = []
            
            for item in batch:
                datapoint_id = item["datapoint_id"]
                img_url = item["image_url"]
                
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    response = requests.get(img_url, timeout=15, headers=headers)
                    
                    if response.status_code != 200:
                        logger.warning(f"Failed to download {datapoint_id}: HTTP {response.status_code}")
                        failed_count += 1
                        continue
                    
                    image_bytes = response.content
                    
                    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB
                        logger.warning(f"Image too large {datapoint_id}: {len(image_bytes)} bytes")
                        failed_count += 1
                        continue
                    
                    # Tạo embedding với delay để tránh quota
                    emb = _vs().create_image_embedding_from_bytes(image_bytes)
                    if emb:
                        batch_embeddings.append({
                            "datapoint_id": datapoint_id,
                            "product_id": item["product_id"],
                            "embedding": emb,
                            "image_url": img_url,
                            "position": item["position"],
                            "product_name": item["product_name"]
                        })
                    else:
                        failed_count += 1
                    
                    # Delay giữa các request để tránh quota
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Failed to embed {datapoint_id}: {e}")
                    failed_count += 1
                    continue

            # Lưu vào MongoDB và Vertex
            if batch_embeddings:
                now = datetime.now()
                docs = []
                pairs_for_upsert = []
                
                for item in batch_embeddings:
                    docs.append({
                        "datapoint_id": item["datapoint_id"],
                        "product_id": item["product_id"],
                        "embedding": item["embedding"],
                        "image_url": item["image_url"],
                        "position": item["position"],
                        "product_name": item["product_name"],
                        "created_at": now,
                    })
                    # Sử dụng datapoint_id thay vì product_id
                    pairs_for_upsert.append((item["datapoint_id"], item["embedding"]))

                if docs:
                    try:
                        image_embeddings_col.insert_many(docs, ordered=False)
                    except Exception as me:
                        logger.error(f"MongoDB insert failed: {me}")

                if pairs_for_upsert:
                    try:
                        _vs().upsert_image_vectors(pairs_for_upsert)
                        total_upsert += len(pairs_for_upsert)
                        logger.info(f"Upserted {len(pairs_for_upsert)} image vectors, total={total_upsert}")
                    except Exception as ve:
                        logger.error(f"Failed to upsert image vectors: {ve}")

            # Tăng delay giữa các batch
            time.sleep(2.0)  # Tăng từ 1.0 lên 2.0

        return jsonify({
            "success": True,
            "total_images_indexed": total_upsert,
            "failed_count": failed_count,
            "message": f"Indexed {total_upsert} images from all products"
        }), 200

    except Exception as e:
        logger.error(f"Rebuild multi-image index failed: {e}")
        return jsonify({"error": str(e)}), 500


@multi_image_index_bp.route("/search-by-image-multi", methods=["POST"])
def search_by_image_multi():
    """
    Tìm kiếm với multi-image support
    
    Kết quả sẽ group theo product_id và lấy match tốt nhất
    """
    try:
        from flask import request
        import base64
        from bson import ObjectId
        
        top_k = 20  # Lấy nhiều hơn để có thể group
        image_bytes = None
        query_emb = None
        final_top_k = 5  # Default
        
        if request.content_type and 'multipart/form-data' in request.content_type:
            if 'image' not in request.files:
                return jsonify({"error": "No image file provided"}), 400
            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "Empty filename"}), 400
            image_bytes = file.read()
            top_k = int(request.form.get('top_k', 20))
            final_top_k = int(request.form.get('top_k', 5))
        else:
            data = request.get_json(silent=True) or {}
            top_k = int(data.get('top_k', 20))
            final_top_k = int(data.get('top_k', 5))
            if 'image_base64' in data:
                image_bytes = base64.b64decode(
                    data['image_base64'].split(',')[1] 
                    if ',' in data['image_base64'] 
                    else data['image_base64']
                )
            elif 'gcs_uri' in data:
                query_emb = _vs().create_image_embedding_from_gcs(data['gcs_uri'])
            else:
                return jsonify({"error": "No image provided"}), 400

        if image_bytes and not query_emb:
            query_emb = _vs().create_image_embedding_from_bytes(image_bytes)
        
        if not query_emb:
            return jsonify({"error": "Failed to create image embedding"}), 500

        # Tìm kiếm trong image index
        neighbors = _vs().find_image_neighbors(query_emb, k=top_k)

        # Group theo product_id và lấy best match
        mongo = current_app.config["MONGODB_SERVICE"]
        products_col = mongo.db["products"]
        image_embeddings_col = mongo.db["product_image_embeddings"]
        
        product_scores = {}  # {product_id: (best_score, matched_image_url, position)}
        
        for datapoint_id, dist in neighbors:
            # Parse datapoint_id: "product_id_position"
            parts = datapoint_id.rsplit('_', 1)
            if len(parts) != 2:
                continue
            
            product_id = parts[0]
            
            # Lưu best match (distance nhỏ nhất)
            if product_id not in product_scores or dist < product_scores[product_id][0]:
                # Lấy image info
                img_doc = image_embeddings_col.find_one({"datapoint_id": datapoint_id})
                if img_doc:
                    product_scores[product_id] = (
                        dist,
                        img_doc.get("image_url"),
                        img_doc.get("position", 0)
                    )

        # Sắp xếp theo score và lấy top products
        sorted_products = sorted(product_scores.items(), key=lambda x: x[1][0])
        final_top_k = int(request.form.get('top_k', 5) if request.content_type and 'multipart/form-data' in request.content_type else data.get('top_k', 5))
        sorted_products = sorted_products[:final_top_k]

        # Lấy thông tin product
        results = []
        for product_id, (score, matched_url, position) in sorted_products:
            try:
                oid = ObjectId(product_id) if ObjectId.is_valid(product_id) else product_id
                product = products_col.find_one({"_id": oid})
            except:
                product = products_col.find_one({"_id": product_id})
            
            if not product:
                continue

            if "_id" in product:
                product["_id"] = str(product["_id"])

            results.append({
                "product": product,
                "similarity_score": float(score),
                "matched_image": {
                    "url": matched_url,
                    "position": position
                }
            })

        return jsonify({
            "success": True,
            "search_type": "multi_image",
            "total_results": len(results),
            "results": results
        }), 200

    except Exception as e:
        logger.exception("Multi-image search failed")
        return jsonify({"error": str(e)}), 500