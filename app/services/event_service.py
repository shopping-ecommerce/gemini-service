import logging
from typing import Dict, List, Optional
from datetime import datetime
from bson import ObjectId

# from app.routes.recommend import _weighted_mean

logger = logging.getLogger(__name__)

class EventService:
    """
    Service để track và quản lý user events (view, cart, purchase, wishlist)
    """
    def __init__(self, mongodb_service):
        self.mongodb = mongodb_service
        self.collection_name = "events"
    
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
            event_type: Loại event: 'view', 'cart', 'purchase', 'wishlist'
            metadata: Thông tin bổ sung (price, quantity, etc.)
        
        Returns:
            Created event document
        """
        try:
            if event_type not in ['view', 'cart', 'purchase', 'wishlist']:
                raise ValueError(f"Invalid event_type: {event_type}")
            
            event = {
                "userId": user_id,
                "productId": product_id,
                "type": event_type,
                "ts": datetime.utcnow(),
                "metadata": metadata or {}
            }
            
            collection = self.mongodb.db[self.collection_name]
            result = collection.insert_one(event)
            
            event["_id"] = str(result.inserted_id)
            
            logger.info(f"✓ Tracked {event_type} event: user={user_id}, product={product_id}")
            return event
            
        except Exception as e:
            logger.error(f"Error tracking event: {e}")
            raise
    
    def get_user_events(
        self, 
        user_id: str, 
        event_type: Optional[str] = None,
        limit: int = 200
    ) -> List[Dict]:
        """
        Lấy lịch sử events của user
        
        Args:
            user_id: ID của user
            event_type: Filter theo loại event (optional)
            limit: Số events tối đa
        
        Returns:
            List of events
        """
        try:
            collection = self.mongodb.db[self.collection_name]
            
            query = {"userId": user_id}
            if event_type:
                query["type"] = event_type
            
            events = list(
                collection.find(query)
                .sort("ts", -1)
                .limit(limit)
            )
            
            # Convert ObjectId to string
            for event in events:
                if "_id" in event:
                    event["_id"] = str(event["_id"])
            
            logger.info(f"Retrieved {len(events)} events for user {user_id}")
            return events
            
        except Exception as e:
            logger.error(f"Error getting user events: {e}")
            raise
    
    def get_product_events(
        self, 
        product_id: str,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Lấy events của một product"""
        try:
            collection = self.mongodb.db[self.collection_name]
            
            query = {"productId": product_id}
            if event_type:
                query["type"] = event_type
            
            events = list(
                collection.find(query)
                .sort("ts", -1)
                .limit(limit)
            )
            
            for event in events:
                if "_id" in event:
                    event["_id"] = str(event["_id"])
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting product events: {e}")
            raise
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Thống kê events của user"""
        try:
            collection = self.mongodb.db[self.collection_name]
            
            # Aggregate by event type
            pipeline = [
                {"$match": {"userId": user_id}},
                {"$group": {
                    "_id": "$type",
                    "count": {"$sum": 1}
                }}
            ]
            
            results = list(collection.aggregate(pipeline))
            
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
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            raise
    
    def batch_track_events(self, events: List[Dict]) -> Dict:
        """
        Track nhiều events cùng lúc
        
        Args:
            events: List of event dicts with keys: user_id, product_id, type, metadata
        
        Returns:
            Summary of inserted events
        """
        try:
            collection = self.mongodb.db[self.collection_name]
            
            docs = []
            for event in events:
                docs.append({
                    "userId": event.get("user_id"),
                    "productId": event.get("product_id"),
                    "type": event.get("type"),
                    "ts": event.get("ts", datetime.utcnow()),
                    "metadata": event.get("metadata", {})
                })
            
            result = collection.insert_many(docs)
            
            logger.info(f"✓ Batch tracked {len(result.inserted_ids)} events")
            
            return {
                "inserted_count": len(result.inserted_ids),
                "inserted_ids": [str(id) for id in result.inserted_ids]
            }
            
        except Exception as e:
            logger.error(f"Error batch tracking events: {e}")
            raise
    # def build_user_profile_vector(self, user_id: str, lookback_days=90, weights=None):
    #     from datetime import datetime, timedelta
    #     weights = weights or {"purchase":3.0, "cart":2.0, "wishlist":1.5, "view":1.0}

    #     events_col = self.mongodb_service.db["events"]
    #     emb_col = self.mongodb_service.db["product_embeddings"]

    #     since = datetime.utcnow() - timedelta(days=lookback_days)
    #     user_events = list(events_col.find({"user_id": user_id, "timestamp": {"$gte": since}})
    #                                   .sort("timestamp", -1).limit(500))
    #     vecs, vws = [], []
    #     for ev in user_events:
    #         pid = str(ev.get("product_id") or "")
    #         et = (ev.get("type") or "").lower()
    #         w = float(weights.get(et, weights.get("view", 1.0)))
    #         emb_doc = emb_col.find_one({"product_id": pid}, {"embedding": 1})
    #         if emb_doc and emb_doc.get("embedding"):
    #             vecs.append(emb_doc["embedding"])
    #             vws.append(w)

    #     return _weighted_mean(vecs, vws)  # bạn có thể copy _weighted_mean vào đây