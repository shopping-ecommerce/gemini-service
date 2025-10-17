import logging
from pymongo import MongoClient
from bson import ObjectId
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MongoDBService:
    def __init__(self, uri: str, database_name: str = "product_db"):
        """
        Khởi tạo MongoDB connection
        
        Args:
            uri: MongoDB connection string
            database_name: Tên database
        """
        try:
            self.client = MongoClient(uri)
            self.db = self.client[database_name]

            # Test connection
            self.client.server_info()
            logger.info(f"✓ Connected to MongoDB: {database_name}")

            # Cache tên danh mục để giảm truy vấn lặp
            self._category_cache: Dict[str, Optional[str]] = {}

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def get_category_name_by_id(self, category_id: str) -> Optional[str]:
        """
        Trả về tên danh mục theo id.
        Hỗ trợ:
          - _id là string
          - _id là ObjectId
          - field 'id'
        Có cache để giảm I/O.
        """
        if not category_id:
            return None

        if category_id in self._category_cache:
            return self._category_cache[category_id]

        try:
            coll = self.db["categories"]
            doc = None

            # 1) Thử _id = string (Spring Data thường lưu id là String)
            doc = coll.find_one({"_id": category_id})

            # 2) Nếu không có, thử _id = ObjectId
            if not doc and ObjectId.is_valid(category_id):
                try:
                    doc = coll.find_one({"_id": ObjectId(category_id)})
                except Exception:
                    doc = None

            # 3) Cuối cùng thử field 'id'
            if not doc:
                doc = coll.find_one({"id": category_id})

            name = None
            if doc:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                raw = doc.get("name")
                if isinstance(raw, str):
                    name = raw.strip() or None

            self._category_cache[category_id] = name
            return name

        except Exception as e:
            logger.warning(f"Could not load category name for {category_id}: {e}")
            self._category_cache[category_id] = None
            return None

    def get_all_products(self, status: str = "AVAILABLE") -> List[Dict]:
        """
        Lấy tất cả products từ MongoDB
        
        Args:
            status: Filter theo status (default: AVAILABLE)
            
        Returns:
            List of product documents
        """
        try:
            collection = self.db["products"]

            # Query với filter
            query: Dict = {}
            if status:
                query["status"] = status

            products = list(collection.find(query))

            # Convert ObjectId to string và đảm bảo có field 'id'
            for product in products:
                if "_id" in product:
                    product["id"] = str(product["_id"])
                    product["_id"] = str(product["_id"])

            logger.info(f"Retrieved {len(products)} products from MongoDB")
            return products

        except Exception as e:
            logger.error(f"Error fetching products: {e}")
            raise

    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """
        Lấy 1 product theo ID
        Hỗ trợ cả ObjectId và string id
        """
        try:
            collection = self.db["products"]
            product = None

            # Thử tìm bằng ObjectId trước
            try:
                if ObjectId.is_valid(product_id):
                    product = collection.find_one({"_id": ObjectId(product_id)})
            except Exception:
                product = None

            # Nếu không tìm thấy, thử tìm bằng field 'id' (nếu có)
            if not product:
                product = collection.find_one({"id": product_id})

            # Convert ObjectId to string
            if product:
                if "_id" in product:
                    product["id"] = str(product["_id"])
                    product["_id"] = str(product["_id"])

            return product

        except Exception as e:
            logger.error(f"Error fetching product {product_id}: {e}")
            raise

    def get_products_paginated(self, page: int = 1, page_size: int = 100, status: str = "AVAILABLE") -> Dict:
        """
        Lấy products với pagination
        
        Returns:
            {
                "products": [...],
                "total": 1000,
                "page": 1,
                "page_size": 100,
                "total_pages": 10
            }
        """
        try:
            collection = self.db["products"]

            query: Dict = {}
            if status:
                query["status"] = status

            # Count total
            total = collection.count_documents(query)

            # Get paginated results
            skip = (page - 1) * page_size
            products = list(collection.find(query).skip(skip).limit(page_size))

            # Convert ObjectId và thêm field 'id'
            for product in products:
                if "_id" in product:
                    product["id"] = str(product["_id"])
                    product["_id"] = str(product["_id"])

            return {
                "products": products,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size
            }

        except Exception as e:
            logger.error(f"Error in pagination: {e}")
            raise

    def close(self):
        """Đóng connection"""
        if hasattr(self, "client") and self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
