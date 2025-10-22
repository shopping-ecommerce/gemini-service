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
    
    Rate limit: 10 requests/minute cho multimodalembedding@001
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

        # RATE LIMITING: 10 requests/minute = 1 request mỗi 6 giây
        REQUESTS_PER_MINUTE = 8  # Để an toàn, chỉ dùng 8/10 quota
        SECONDS_PER_REQUEST = 60.0 / REQUESTS_PER_MINUTE  # = 7.5 giây/request
        
        batch_size = 3  # Batch nhỏ để dễ quản lý
        total_upsert = 0
        failed_count = 0
        request_count = 0
        minute_start = time.time()

        for batch_idx, batch in enumerate(_chunks(to_embed, batch_size)):
            batch_embeddings = []
            
            for item in batch:
                datapoint_id = item["datapoint_id"]
                img_url = item["image_url"]
                
                # Rate limiting: Đợi nếu đã dùng hết quota trong phút này
                if request_count >= REQUESTS_PER_MINUTE:
                    elapsed = time.time() - minute_start
                    if elapsed < 60:
                        wait_time = 60 - elapsed + 1  # +1 giây để chắc chắn
                        logger.info(f"Rate limit reached. Waiting {wait_time:.1f}s before next batch...")
                        time.sleep(wait_time)
                    # Reset counter
                    request_count = 0
                    minute_start = time.time()
                
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
                    
                    # Tạo embedding - đây là 1 request tính vào quota
                    emb = _vs().create_image_embedding_from_bytes(image_bytes)
                    request_count += 1  # Đếm request
                    
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
                    
                    # Delay giữa các request
                    time.sleep(SECONDS_PER_REQUEST)
                    
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
                        logger.info(f"Upserted {len(pairs_for_upsert)} vectors, total={total_upsert}/{len(to_embed)} ({request_count} requests this minute)")
                    except Exception as ve:
                        logger.error(f"Failed to upsert image vectors: {ve}")

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
    
    Distance nhỏ = Similarity cao = Giống nhau
    """
    try:
        from flask import request
        import base64
        from bson import ObjectId
        import math
        
        image_bytes = None
        query_emb = None
        final_top_k = 5  # Số products cần trả về
        data = {}
        
        content_type = request.content_type or ""
        is_multipart = "multipart/form-data" in content_type
        
        # Parse request
        if is_multipart:
            if "image" not in request.files:
                return jsonify({"error": "No image file provided"}), 400
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"error": "Empty filename"}), 400
            image_bytes = file.read()
            final_top_k = int(request.form.get("top_k", 5))
        else:
            data = request.get_json(silent=True) or {}
            final_top_k = int(data.get("top_k", 5))
            
            if "image_base64" in data:
                image_bytes = base64.b64decode(
                    data["image_base64"].split(",")[1]
                    if "," in data["image_base64"]
                    else data["image_base64"]
                )
            elif "gcs_uri" in data:
                query_emb = _vs().create_image_embedding_from_url(data["gcs_uri"])
            else:
                return jsonify({"error": "No image provided"}), 400

        # Optional: ngưỡng lọc similarity
        try:
            min_similarity = float(request.form.get("min_similarity", 0.0)) if is_multipart else float(data.get("min_similarity", 0.0))
        except Exception:
            min_similarity = 0.0

        # Tạo embedding từ ảnh
        if image_bytes and not query_emb:
            query_emb = _vs().create_image_embedding_from_bytes(image_bytes)
        
        if not query_emb:
            return jsonify({"error": "Failed to create image embedding"}), 500

        # Tìm kiếm trong image index
        search_k = final_top_k * 10  # Lấy dư để đủ sản phẩm sau khi group
        neighbors = _vs().find_image_neighbors(query_emb, k=search_k)
        
        if not neighbors:
            return jsonify({
                "success": True,
                "search_type": "multi_image",
                "total_results": 0,
                "results": [],
                "message": "No similar images found"
            }), 200

        # Group theo product_id và lấy best match (distance nhỏ nhất)
        mongo = current_app.config["MONGODB_SERVICE"]
        products_col = mongo.db["products"]
        image_embeddings_col = mongo.db["product_image_embeddings"]
        
        # {product_id: {"distance": float, "matched_image_url": str, "position": int, "datapoint_id": str, "embedding": list}}
        product_scores = {}
        
        for datapoint_id, distance in neighbors:
            # Parse datapoint_id: "product_id_position"
            parts = datapoint_id.rsplit("_", 1)
            if len(parts) != 2:
                logger.warning(f"Invalid datapoint_id format: {datapoint_id}")
                continue
            
            product_id = parts[0]
            
            # Lưu best match (distance nhỏ nhất)
            if product_id not in product_scores or distance < product_scores[product_id]["distance"]:
                img_doc = image_embeddings_col.find_one(
                    {"datapoint_id": datapoint_id},
                    {"image_url": 1, "position": 1, "embedding": 1}
                )
                if img_doc:
                    product_scores[product_id] = {
                        "distance": float(distance),
                        "matched_image_url": img_doc.get("image_url"),
                        "position": img_doc.get("position", 0),
                        "datapoint_id": datapoint_id,
                        "embedding": img_doc.get("embedding"),
                    }

        if not product_scores:
            return jsonify({
                "success": True,
                "search_type": "multi_image",
                "total_results": 0,
                "results": [],
                "message": "No products found"
            }), 200

        # Sắp xếp theo distance (nhỏ nhất = tốt nhất) và lấy top N
        sorted_products = sorted(product_scores.items(), key=lambda x: x[1]["distance"])[:final_top_k]
        
        logger.info(f"Search: found {len(product_scores)} unique products, returning top {len(sorted_products)}")

        # Batch query products từ MongoDB
        product_ids = [pid for pid, _ in sorted_products]
        oids = []
        for pid in product_ids:
            try:
                oids.append(ObjectId(pid) if ObjectId.is_valid(pid) else pid)
            except Exception:
                oids.append(pid)
        
        products_dict = {str(p["_id"]): p for p in products_col.find({"_id": {"$in": oids}})}

        # Helper tính cosine
        def _cosine(a, b):
            try:
                n = min(len(a), len(b))
                if n == 0:
                    return 0.0
                dot = 0.0
                sa = 0.0
                sb = 0.0
                for i in range(n):
                    ai = float(a[i])
                    bi = float(b[i])
                    dot += ai * bi
                    sa += ai * ai
                    sb += bi * bi
                if sa == 0.0 or sb == 0.0:
                    return 0.0
                return dot / (math.sqrt(sa) * math.sqrt(sb))
            except Exception:
                return 0.0

        # Build results
        results = []
        for product_id, info in sorted_products:
            product = products_dict.get(product_id)
            if not product:
                logger.warning(f"Product not found: {product_id}")
                continue

            product["_id"] = str(product["_id"])
            distance = float(info["distance"])
            matched_url = info["matched_image_url"]
            position = info["position"]
            emb = info.get("embedding")

            # Ưu tiên tính cosine similarity trực tiếp từ embedding
            if emb:
                cos = _cosine(query_emb, emb)
                similarity = (cos + 1.0) / 2.0  # map [-1,1] -> [0,1]
            else:
                # Fallback nếu không có embedding: giả định distance là cosine distance trong [0,2]
                similarity = 1.0 - (distance / 2.0)

            # Clamp [0,1]
            if similarity < 0.0:
                similarity = 0.0
            elif similarity > 1.0:
                similarity = 1.0

            # Lọc theo ngưỡng
            if similarity < min_similarity:
                continue

            results.append({
                "product": product,
                "similarity_score": round(similarity, 4),
                "distance": round(distance, 4),
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