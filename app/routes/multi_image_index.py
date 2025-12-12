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


@multi_image_index_bp.route("/index-single-product-images", methods=["POST"])
def index_single_product_images():
    """
    Index tất cả hình ảnh của MỘT product cụ thể
    
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
        image_embeddings_col = mongo.db["product_image_embeddings"]
        
        # Validate product exists
        try:
            oid = ObjectId(product_id) if ObjectId.is_valid(product_id) else product_id
        except Exception:
            oid = product_id
        
        product = products_col.find_one({"_id": oid})
        if not product:
            return jsonify({"error": f"Product {product_id} not found"}), 404
        
        pid = str(product["_id"])
        images = product.get("images", [])
        
        if not images:
            return jsonify({
                "success": True,
                "product_id": pid,
                "indexed_count": 0,
                "message": "Product has no images"
            }), 200
        
        # Nếu force_reindex, xóa embeddings cũ của product này
        if force_reindex:
            old_datapoints = [
                doc.get("datapoint_id") 
                for doc in image_embeddings_col.find({"product_id": pid}, {"datapoint_id": 1})
            ]
            old_datapoints = [did for did in old_datapoints if did]
            
            if old_datapoints:
                image_embeddings_col.delete_many({"product_id": pid})
                try:
                    _vs().remove_image_vectors(old_datapoints)
                    logger.info(f"Removed {len(old_datapoints)} old image vectors for product {pid}")
                except Exception as ve:
                    logger.warning(f"Could not remove old image vectors: {ve}")
        
        # Chuẩn bị danh sách ảnh cần embed
        to_embed = []
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
            
            # Skip nếu đã tồn tại (khi không force_reindex)
            if not force_reindex:
                existing = image_embeddings_col.find_one({"datapoint_id": datapoint_id})
                if existing:
                    logger.info(f"Skipping existing datapoint {datapoint_id}")
                    continue
            
            to_embed.append({
                "datapoint_id": datapoint_id,
                "product_id": pid,
                "image_url": url,
                "position": position,
                "product_name": product.get("name", "")
            })
        
        if not to_embed:
            return jsonify({
                "success": True,
                "product_id": pid,
                "indexed_count": 0,
                "message": "No new images to index"
            }), 200
        
        # Rate limiting settings
        REQUESTS_PER_MINUTE = 8
        SECONDS_PER_REQUEST = 60.0 / REQUESTS_PER_MINUTE
        
        indexed_count = 0
        failed_count = 0
        
        for item in to_embed:
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
                
                if len(image_bytes) > 10 * 1024 * 1024:
                    logger.warning(f"Image too large {datapoint_id}: {len(image_bytes)} bytes")
                    failed_count += 1
                    continue
                
                # Tạo embedding
                emb = _vs().create_image_embedding_from_bytes(image_bytes)
                
                if emb:
                    now = datetime.now()
                    doc = {
                        "datapoint_id": datapoint_id,
                        "product_id": item["product_id"],
                        "embedding": emb,
                        "image_url": img_url,
                        "position": item["position"],
                        "product_name": item["product_name"],
                        "created_at": now,
                    }
                    
                    image_embeddings_col.insert_one(doc)
                    _vs().upsert_image_vector(datapoint_id, emb)
                    
                    indexed_count += 1
                    logger.info(f"Indexed {datapoint_id} for product {pid}")
                else:
                    failed_count += 1
                
                # Delay để tuân thủ rate limit
                time.sleep(SECONDS_PER_REQUEST)
                
            except Exception as e:
                logger.warning(f"Failed to embed {datapoint_id}: {e}")
                failed_count += 1
                continue
        
        return jsonify({
            "success": True,
            "product_id": pid,
            "indexed_count": indexed_count,
            "failed_count": failed_count,
            "message": f"Indexed {indexed_count} images for product {pid}"
        }), 200
        
    except Exception as e:
        logger.error(f"Index single product images failed: {e}")
        return jsonify({"error": str(e)}), 500


@multi_image_index_bp.route("/remove-product-images", methods=["POST"])
def remove_product_images():
    """
    Xóa tất cả image embeddings của MỘT product
    
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
        image_embeddings_col = mongo.db["product_image_embeddings"]
        
        # Lấy danh sách datapoint_ids cần xóa
        datapoints = [
            doc.get("datapoint_id") 
            for doc in image_embeddings_col.find({"product_id": product_id}, {"datapoint_id": 1})
        ]
        datapoints = [did for did in datapoints if did]
        
        if not datapoints:
            return jsonify({
                "success": True,
                "product_id": product_id,
                "removed_count": 0,
                "message": "No embeddings found for this product"
            }), 200
        
        # Xóa từ MongoDB
        result = image_embeddings_col.delete_many({"product_id": product_id})
        
        # Xóa từ Vertex AI
        try:
            _vs().remove_image_vectors(datapoints)
            logger.info(f"Removed {len(datapoints)} image vectors for product {product_id}")
        except Exception as ve:
            logger.warning(f"Could not remove vectors from Vertex: {ve}")
        
        return jsonify({
            "success": True,
            "product_id": product_id,
            "removed_count": result.deleted_count,
            "message": f"Removed {result.deleted_count} image embeddings"
        }), 200
        
    except Exception as e:
        logger.error(f"Remove product images failed: {e}")
        return jsonify({"error": str(e)}), 500


@multi_image_index_bp.route("/upsert-single-image", methods=["POST"])
def upsert_single_image():
    """
    Thêm hoặc cập nhật embedding cho MỘT ảnh cụ thể
    
    Request body (multipart/form-data hoặc JSON):
    - product_id: ID của product
    - position: Vị trí của ảnh (số nguyên)
    - image_url: URL của ảnh (nếu dùng JSON)
    - image: File ảnh (nếu dùng multipart)
    """
    try:
        from flask import request
        
        content_type = request.content_type or ""
        is_multipart = "multipart/form-data" in content_type
        
        product_id = None
        position = None
        image_url = None
        image_bytes = None
        
        if is_multipart:
            product_id = request.form.get("product_id")
            try:
                position = int(request.form.get("position", 0))
            except Exception:
                position = 0
            
            if "image" in request.files:
                file = request.files["image"]
                if file.filename != "":
                    image_bytes = file.read()
        else:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
            
            product_id = data.get("product_id")
            try:
                position = int(data.get("position", 0))
            except Exception:
                position = 0
            image_url = data.get("image_url")
        
        if not product_id:
            return jsonify({"error": "product_id is required"}), 400
        
        if not image_url and not image_bytes:
            return jsonify({"error": "Either image_url or image file is required"}), 400
        
        datapoint_id = f"{product_id}_{position}"
        
        mongo = current_app.config["MONGODB_SERVICE"]
        products_col = mongo.db["products"]
        image_embeddings_col = mongo.db["product_image_embeddings"]
        
        # Validate product exists
        from bson import ObjectId
        try:
            oid = ObjectId(product_id) if ObjectId.is_valid(product_id) else product_id
        except Exception:
            oid = product_id
        
        product = products_col.find_one({"_id": oid}, {"name": 1})
        if not product:
            return jsonify({"error": f"Product {product_id} not found"}), 404
        
        product_name = product.get("name", "")
        
        # Download image nếu cần
        if image_url and not image_bytes:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(image_url, timeout=15, headers=headers)
                
                if response.status_code != 200:
                    return jsonify({"error": f"Failed to download image: HTTP {response.status_code}"}), 400
                
                image_bytes = response.content
                
            except Exception as e:
                return jsonify({"error": f"Failed to download image: {str(e)}"}), 400
        
        if len(image_bytes) > 10 * 1024 * 1024:
            return jsonify({"error": "Image too large (max 10MB)"}), 400
        
        # Tạo embedding
        emb = _vs().create_image_embedding_from_bytes(image_bytes)
        
        if not emb:
            return jsonify({"error": "Failed to create image embedding"}), 500
        
        # Upsert vào MongoDB
        now = datetime.now()
        doc = {
            "datapoint_id": datapoint_id,
            "product_id": product_id,
            "embedding": emb,
            "image_url": image_url or f"uploaded_{datapoint_id}",
            "position": position,
            "product_name": product_name,
            "created_at": now,
            "updated_at": now,
        }
        
        image_embeddings_col.replace_one(
            {"datapoint_id": datapoint_id},
            doc,
            upsert=True
        )
        
        # Upsert vào Vertex AI
        _vs().upsert_image_vector(datapoint_id, emb)
        
        logger.info(f"Upserted image embedding {datapoint_id} for product {product_id}")
        
        return jsonify({
            "success": True,
            "datapoint_id": datapoint_id,
            "product_id": product_id,
            "position": position,
            "message": f"Successfully upserted image embedding"
        }), 200
        
    except Exception as e:
        logger.error(f"Upsert single image failed: {e}")
        return jsonify({"error": str(e)}), 500


@multi_image_index_bp.route("/remove-single-image", methods=["POST"])
def remove_single_image():
    """
    Xóa embedding của MỘT ảnh cụ thể
    
    Request body:
    {
        "product_id": "67123abc...",
        "position": 0
    }
    hoặc
    {
        "datapoint_id": "67123abc..._0"
    }
    """
    try:
        from flask import request
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        datapoint_id = data.get("datapoint_id")
        
        if not datapoint_id:
            product_id = data.get("product_id")
            position = data.get("position")
            
            if not product_id or position is None:
                return jsonify({"error": "Either datapoint_id or (product_id + position) is required"}), 400
            
            datapoint_id = f"{product_id}_{position}"
        
        mongo = current_app.config["MONGODB_SERVICE"]
        image_embeddings_col = mongo.db["product_image_embeddings"]
        
        # Kiểm tra tồn tại
        existing = image_embeddings_col.find_one({"datapoint_id": datapoint_id})
        if not existing:
            return jsonify({
                "success": True,
                "datapoint_id": datapoint_id,
                "message": "Embedding not found (already removed)"
            }), 200
        
        # Xóa từ MongoDB
        image_embeddings_col.delete_one({"datapoint_id": datapoint_id})
        
        # Xóa từ Vertex AI
        try:
            _vs().remove_image_vectors([datapoint_id])
            logger.info(f"Removed image vector {datapoint_id}")
        except Exception as ve:
            logger.warning(f"Could not remove vector from Vertex: {ve}")
        
        return jsonify({
            "success": True,
            "datapoint_id": datapoint_id,
            "message": "Successfully removed image embedding"
        }), 200
        
    except Exception as e:
        logger.error(f"Remove single image failed: {e}")
        return jsonify({"error": str(e)}), 500

@multi_image_index_bp.route("/search-by-image-multi", methods=["POST"])
def search_by_image_multi():
    """
    Tìm kiếm với multi-image support

    Quy trình:
    1) Tạo embedding cho ảnh truy vấn.
    2) Gọi ANN với candidate_k cố định (ổn định recall, không phụ thuộc top_k).
    3) Lấy danh sách product ứng viên từ tập neighbor.
    4) Re-rank chính xác bằng cosine trên tối đa per_product_rerank ảnh/ product.
    5) Chuẩn hóa similarity về [0..1], lọc min_similarity và trả về top_k ổn định.
    """
    try:
        from flask import request
        import base64
        from bson import ObjectId
        import math

        # Parse đầu vào
        image_bytes = None
        query_emb = None
        final_top_k = 5
        data = {}

        content_type = request.content_type or ""
        is_multipart = "multipart/form-data" in content_type

        if is_multipart:
            if "image" not in request.files:
                return jsonify({"error": "No image file provided"}), 400
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"error": "Empty filename"}), 400
            image_bytes = file.read()
            try:
                final_top_k = int(request.form.get("top_k", 5))
            except Exception:
                final_top_k = 5
        else:
            data = request.get_json(silent=True) or {}
            try:
                final_top_k = int(data.get("top_k", 5))
            except Exception:
                final_top_k = 5

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

        # Tham số nâng cao
        default_candidate_k = 300
        max_candidate_k = 1000
        default_per_product_rerank = 8

        try:
            candidate_k = int(request.form.get("candidate_k", default_candidate_k)) if is_multipart else int(data.get("candidate_k", default_candidate_k))
        except Exception:
            candidate_k = default_candidate_k
        candidate_k = max(50, min(candidate_k, max_candidate_k))

        try:
            per_product_rerank = int(request.form.get("per_product_rerank", default_per_product_rerank)) if is_multipart else int(data.get("per_product_rerank", default_per_product_rerank))
        except Exception:
            per_product_rerank = default_per_product_rerank
        per_product_rerank = max(1, min(per_product_rerank, 16))

        # Optional: ngưỡng lọc similarity
        try:
            min_similarity = float(request.form.get("min_similarity", 0.0)) if is_multipart else float(data.get("min_similarity", 0.0))
        except Exception:
            min_similarity = 0.0

        # Tạo embedding query
        if image_bytes and not query_emb:
            query_emb = _vs().create_image_embedding_from_bytes(image_bytes)
        if not query_emb:
            return jsonify({"error": "Failed to create image embedding"}), 500

        # ANN search với candidate_k cố định (không phụ thuộc final_top_k)
        neighbors = _vs().find_image_neighbors(query_emb, k=candidate_k)
        if not neighbors:
            return jsonify({
                "success": True,
                "search_type": "multi_image",
                "total_results": 0,
                "results": [],
                "message": "No similar images found"
            }), 200

        # Ổn định thứ tự neighbor (deterministic tie-break)
        neighbors = sorted(
            ((str(dpid), float(dist)) for dpid, dist in neighbors),
            key=lambda x: (x[1], x[0])
        )

        # Group product ứng viên theo tập neighbor
        mongo = current_app.config["MONGODB_SERVICE"]
        products_col = mongo.db["products"]
        image_embeddings_col = mongo.db["product_image_embeddings"]

        candidate_product_ids = []
        seen = set()
        for datapoint_id, _ in neighbors:
            parts = datapoint_id.rsplit("_", 1)
            if len(parts) != 2:
                logger.warning(f"Invalid datapoint_id format: {datapoint_id}")
                continue
            pid = parts[0]
            if pid not in seen:
                seen.add(pid)
                candidate_product_ids.append(pid)

        if not candidate_product_ids:
            return jsonify({
                "success": True,
                "search_type": "multi_image",
                "total_results": 0,
                "results": [],
                "message": "No products found"
            }), 200

        # Helper: cosine
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
                return dot / ((sa ** 0.5) * (sb ** 0.5))
            except Exception:
                return 0.0

        # Re-rank chính xác theo cosine trên tối đa per_product_rerank ảnh/product
        product_scores = {}  # {product_id: {"similarity": float, "distance": float, "matched_image_url": str, "position": int, "datapoint_id": str}}
        for pid in candidate_product_ids:
            best = None
            try:
                cursor = image_embeddings_col.find(
                    {"product_id": pid},
                    {"embedding": 1, "image_url": 1, "position": 1, "datapoint_id": 1}
                ).sort("position", 1).limit(per_product_rerank)

                for doc in cursor:
                    emb = doc.get("embedding")
                    if not emb:
                        continue
                    cos = _cosine(query_emb, emb)
                    similarity = (cos + 1.0) / 2.0  # map [-1,1] -> [0,1]
                    # distance từ cosine (nhất quán), trong [0..2]
                    distance = 1.0 - cos
                    cand = {
                        "similarity": float(similarity),
                        "distance": float(distance),
                        "matched_image_url": doc.get("image_url"),
                        "position": doc.get("position", 0),
                        "datapoint_id": doc.get("datapoint_id", "")
                    }
                    if (best is None) or (cand["similarity"] > best["similarity"]) or (
                        cand["similarity"] == best["similarity"] and (cand["distance"] < best["distance"] or cand["datapoint_id"] < best.get("datapoint_id", ""))
                    ):
                        best = cand
            except Exception as e:
                logger.warning(f"Re-rank failed for product {pid}: {e}")

            if best:
                product_scores[pid] = best

        if not product_scores:
            return jsonify({
                "success": True,
                "search_type": "multi_image",
                "total_results": 0,
                "results": [],
                "message": "No products found after rerank"
            }), 200

        # Lọc theo min_similarity và sort ổn định
        filtered = [(pid, info) for pid, info in product_scores.items() if info["similarity"] >= min_similarity]

        sorted_products = sorted(
            filtered,
            key=lambda x: (-x[1]["similarity"], x[1]["distance"], x[1]["datapoint_id"])
        )[:final_top_k]

        logger.info(
            f"Search candidates={len(candidate_product_ids)} reranked={len(product_scores)} "
            f"filtered={len(filtered)} return={len(sorted_products)} "
            f"(candidate_k={candidate_k}, per_product_rerank={per_product_rerank}, top_k={final_top_k})"
        )

        # Batch query products
        product_ids = [pid for pid, _ in sorted_products]
        oids = []
        for pid in product_ids:
            try:
                oids.append(ObjectId(pid) if ObjectId.is_valid(pid) else pid)
            except Exception:
                oids.append(pid)

        products_dict = {str(p["_id"]): p for p in products_col.find({"_id": {"$in": oids}})}

        # Build results
        results = []
        for product_id, info in sorted_products:
            product = products_dict.get(product_id)
            if not product:
                logger.warning(f"Product not found: {product_id}")
                continue

            product["_id"] = str(product["_id"])

            results.append({
                "product": product,
                "similarity_score": round(float(info["similarity"]), 4),
                "distance": round(float(info["distance"]), 4),
                "matched_image": {
                    "url": info["matched_image_url"],
                    "position": info["position"]
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