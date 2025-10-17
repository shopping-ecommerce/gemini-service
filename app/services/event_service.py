# app/services/event_service.py
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from bson import ObjectId

logger = logging.getLogger(__name__)

class EventService:
    """Service quản lý events tracking (view, cart, purchase, wishlist)"""
    
    def __init__(self, mongodb_service):
        """
        Args:
            mongodb_service: Instance của MongoDBService
        """
        self.mongodb = mongodb_service
        self.collection_name = "events"
        self.collection = self.mongodb.db[self.collection_name]
        
        # Tạo indexes để tối ưu query
        self._ensure_indexes()
        logger.info(f"✓ EventService initialized with collection: {self.collection_name}")
    
    def _ensure_indexes(self):
        """Tạo indexes cho collection events"""
        try:
            # Index cho query theo userId
            self.collection.create_index("userId")
            
            # Index cho query theo productId
            self.collection.create_index("productId")
            
            # Compound index cho query userId + type
            self.collection.create_index([("userId", 1), ("type", 1)])
            
            # Index cho timestamp để sort
            self.collection.create_index([("ts", -1)])
            
            logger.info("✓ Event indexes created")
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")
    
    def track_event(
        self, 
        user_id: str, 
        product_id: str, 
        event_type: str, 
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Track một event của user
        
        Args:
            user_id: ID của user
            product_id: ID của product
            event_type: Loại event (view, cart, purchase, wishlist)
            metadata: Metadata bổ sung (price, quantity, etc.)
            
        Returns:
            Dict chứa thông tin event đã tạo
        """
        # Validate event_type
        valid_types = ["view", "cart", "purchase", "wishlist"]
        if event_type not in valid_types:
            raise ValueError(f"event_type phải là một trong: {valid_types}")
        
        # Tạo event document
        event = {
            "userId": user_id,
            "productId": product_id,
            "type": event_type,
            "ts": datetime.utcnow(),
            "metadata": metadata or {}
        }
        
        # Insert vào MongoDB
        result = self.collection.insert_one(event)
        event["_id"] = str(result.inserted_id)
        
        logger.info(f"Event tracked: user={user_id}, product={product_id}, type={event_type}")
        return event
    
    def batch_track_events(self, events: List[Dict]) -> Dict:
        """
        Track nhiều events cùng lúc
        
        Args:
            events: List các event dict với keys: user_id, product_id, type, metadata (optional)
            
        Returns:
            Dict với thông tin kết quả
        """
        if not events:
            return {"success": 0, "failed": 0, "errors": []}
        
        success_count = 0
        failed_count = 0
        errors = []
        
        documents = []
        for idx, event_data in enumerate(events):
            try:
                user_id = event_data.get("user_id")
                product_id = event_data.get("product_id")
                event_type = event_data.get("type")
                metadata = event_data.get("metadata", {})
                
                if not all([user_id, product_id, event_type]):
                    raise ValueError("Missing required fields")
                
                # Validate type
                valid_types = ["view", "cart", "purchase", "wishlist"]
                if event_type not in valid_types:
                    raise ValueError(f"Invalid event type: {event_type}")
                
                documents.append({
                    "userId": user_id,
                    "productId": product_id,
                    "type": event_type,
                    "ts": datetime.utcnow(),
                    "metadata": metadata
                })
                
            except Exception as e:
                failed_count += 1
                errors.append(f"Event {idx}: {str(e)}")
        
        # Bulk insert
        if documents:
            try:
                result = self.collection.insert_many(documents, ordered=False)
                success_count = len(result.inserted_ids)
            except Exception as e:
                logger.error(f"Batch insert failed: {e}")
                failed_count += len(documents)
                errors.append(f"Batch insert error: {str(e)}")
        
        logger.info(f"Batch tracking: {success_count} success, {failed_count} failed")
        return {
            "success": success_count,
            "failed": failed_count,
            "errors": errors
        }
    
    def get_user_events(
        self, 
        user_id: str, 
        event_type: Optional[str] = None, 
        limit: int = 100
    ) -> List[Dict]:
        """
        Lấy danh sách events của user
        
        Args:
            user_id: ID của user
            event_type: Filter theo loại event (optional)
            limit: Số lượng events tối đa
            
        Returns:
            List các event dict
        """
        query = {"userId": user_id}
        
        if event_type:
            query["type"] = event_type
        
        # Sort theo timestamp mới nhất
        events = list(
            self.collection
            .find(query)
            .sort("ts", -1)
            .limit(limit)
        )
        
        # Convert ObjectId to string
        for event in events:
            event["_id"] = str(event["_id"])
        
        logger.debug(f"Found {len(events)} events for user {user_id}")
        return events
    
    def get_user_stats(self, user_id: str) -> Dict:
        """
        Thống kê events của user
        
        Args:
            user_id: ID của user
            
        Returns:
            Dict chứa thống kê
        """
        pipeline = [
            {"$match": {"userId": user_id}},
            {"$group": {
                "_id": "$type",
                "count": {"$sum": 1}
            }}
        ]
        
        results = list(self.collection.aggregate(pipeline))
        
        stats = {
            "user_id": user_id,
            "total_events": 0,
            "by_type": {}
        }
        
        for result in results:
            event_type = result["_id"]
            count = result["count"]
            stats["by_type"][event_type] = count
            stats["total_events"] += count
        
        # Lấy unique products
        unique_products = self.collection.distinct("productId", {"userId": user_id})
        stats["unique_products"] = len(unique_products)
        
        return stats
    
    def get_product_events(
        self, 
        product_id: str, 
        event_type: Optional[str] = None, 
        limit: int = 100
    ) -> List[Dict]:
        """
        Lấy danh sách events của một product
        
        Args:
            product_id: ID của product
            event_type: Filter theo loại event (optional)
            limit: Số lượng events tối đa
            
        Returns:
            List các event dict
        """
        query = {"productId": product_id}
        
        if event_type:
            query["type"] = event_type
        
        events = list(
            self.collection
            .find(query)
            .sort("ts", -1)
            .limit(limit)
        )
        
        # Convert ObjectId to string
        for event in events:
            event["_id"] = str(event["_id"])
        
        logger.debug(f"Found {len(events)} events for product {product_id}")
        return events
    
    def get_product_stats(self, product_id: str) -> Dict:
        """
        Thống kê events của product
        
        Args:
            product_id: ID của product
            
        Returns:
            Dict chứa thống kê
        """
        pipeline = [
            {"$match": {"productId": product_id}},
            {"$group": {
                "_id": "$type",
                "count": {"$sum": 1}
            }}
        ]
        
        results = list(self.collection.aggregate(pipeline))
        
        stats = {
            "product_id": product_id,
            "total_events": 0,
            "by_type": {}
        }
        
        for result in results:
            event_type = result["_id"]
            count = result["count"]
            stats["by_type"][event_type] = count
            stats["total_events"] += count
        
        # Lấy unique users
        unique_users = self.collection.distinct("userId", {"productId": product_id})
        stats["unique_users"] = len(unique_users)
        
        return stats
    
    def delete_user_events(self, user_id: str) -> int:
        """
        Xóa tất cả events của user (GDPR compliance)
        
        Args:
            user_id: ID của user
            
        Returns:
            Số lượng events đã xóa
        """
        result = self.collection.delete_many({"userId": user_id})
        logger.info(f"Deleted {result.deleted_count} events for user {user_id}")
        return result.deleted_count
    
    def get_popular_products(
        self, 
        event_type: Optional[str] = None, 
        limit: int = 20,
        days: Optional[int] = None
    ) -> List[Dict]:
        """
        Lấy danh sách products phổ biến dựa trên số events
        
        Args:
            event_type: Filter theo loại event (optional)
            limit: Số lượng products
            days: Lọc events trong N ngày gần đây (optional)
            
        Returns:
            List các product với popularity score
        """
        match_stage = {}
        
        if event_type:
            match_stage["type"] = event_type
        
        if days:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            match_stage["ts"] = {"$gte": cutoff_date}
        
        pipeline = [
            {"$match": match_stage} if match_stage else {"$match": {}},
            {"$group": {
                "_id": "$productId",
                "count": {"$sum": 1},
                "unique_users": {"$addToSet": "$userId"}
            }},
            {"$addFields": {
                "unique_user_count": {"$size": "$unique_users"}
            }},
            {"$sort": {"count": -1}},
            {"$limit": limit},
            {"$project": {
                "product_id": "$_id",
                "event_count": "$count",
                "unique_users": "$unique_user_count",
                "_id": 0
            }}
        ]
        
        results = list(self.collection.aggregate(pipeline))
        return results