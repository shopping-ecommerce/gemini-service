# services/vertex_ai_service.py
import logging, time, random
from typing import List, Tuple, Iterable, Optional
from google.api_core.exceptions import ResourceExhausted
import base64

from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.vision_models import MultiModalEmbeddingModel, Image  # ← Thêm này

logger = logging.getLogger(__name__)

class VertexAIService:
    def __init__(
        self,
        project_id: str,
        location: str,
        index_endpoint_id: str,
        deployed_index_id: Optional[str] = None,
        index_id: Optional[str] = None,
        image_index_endpoint_id: Optional[str] = None,  # ← Image index
        image_deployed_index_id: Optional[str] = None,
        image_index_id: Optional[str] = None,
        model_name: str = "gemini-embedding-001",
        image_model_name: str = "multimodalembedding@001",  # ← Image model
    ):
        aiplatform.init(project=project_id, location=location)
        vertexai.init(project=project_id, location=location)

        # Text embedding
        self.embedding_model = TextEmbeddingModel.from_pretrained(model_name)
        self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=index_endpoint_id)
        self.deployed_index_id = deployed_index_id
        self.index = aiplatform.MatchingEngineIndex(index_name=index_id) if index_id else None

        # Image embedding ← MỚI
        self.image_embedding_model = MultiModalEmbeddingModel.from_pretrained(image_model_name)
        self.image_index_endpoint = (
            aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=image_index_endpoint_id) 
            if image_index_endpoint_id else None
        )
        self.image_deployed_index_id = image_deployed_index_id
        self.image_index = (
            aiplatform.MatchingEngineIndex(index_name=image_index_id) 
            if image_index_id else None
        )

        logger.info("✓ VertexAIService initialized (project=%s, location=%s, text_model=%s, image_model=%s)",
                    project_id, location, model_name, image_model_name)

    # ---------- Retry helper ----------
    def _retry(self, fn, *, max_retries=6, base_sleep=0.6):
        for i in range(max_retries):
            try:
                return fn()
            except ResourceExhausted as e:
                sleep = base_sleep * (2 ** i) + random.uniform(0, 0.25)
                logger.warning("429 quota; retry %d in %.2fs: %s", i+1, sleep, e)
                time.sleep(sleep)
        return fn()

    # ---------- Text Embeddings (giữ nguyên) ----------
    def create_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        inp = TextEmbeddingInput(text=text, task_type=task_type)
        return self._retry(lambda: self.embedding_model.get_embeddings([inp])[0].values)

    def create_embeddings_batch(
        self, texts: Iterable[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[List[float]]:
        inputs = [TextEmbeddingInput(text=t, task_type=task_type) for t in texts]
        return self._retry(lambda: [e.values for e in self.embedding_model.get_embeddings(inputs)])

    # ---------- Image Embeddings ← MỚI ----------
    def create_image_embedding_from_url(self, image_url: str, contextual_text: str = None) -> List[float]:
        """
        Tạo embedding từ image URL
        
        Args:
            image_url: URL của ảnh (GCS hoặc public URL)
            contextual_text: Text bổ sung về ảnh (optional, giúp cải thiện độ chính xác)
        """
        try:
            image = Image.load_from_file(image_url) if image_url.startswith("gs://") else None
            if not image:
                # Nếu là HTTP URL, download về
                import requests
                from io import BytesIO
                resp = requests.get(image_url, timeout=10)
                resp.raise_for_status()
                image = Image(image_bytes=resp.content)
            
            embeddings = self._retry(
                lambda: self.image_embedding_model.get_embeddings(
                    image=image,
                    contextual_text=contextual_text
                )
            )
            return embeddings.image_embedding
        except Exception as e:
            logger.error(f"Failed to create image embedding from URL {image_url}: {e}")
            raise

    def create_image_embedding_from_bytes(self, image_bytes: bytes, contextual_text: str = None) -> List[float]:
        """
        Tạo embedding từ image bytes (upload từ client)
        """
        try:
            image = Image(image_bytes=image_bytes)
            embeddings = self._retry(
                lambda: self.image_embedding_model.get_embeddings(
                    image=image,
                    contextual_text=contextual_text
                )
            )
            return embeddings.image_embedding
        except Exception as e:
            logger.error(f"Failed to create image embedding from bytes: {e}")
            raise

    def create_image_embeddings_batch(
        self, 
        image_sources: List[Tuple[str, Optional[str]]]  # [(url/path, contextual_text), ...]
    ) -> List[List[float]]:
        """
        Batch tạo embeddings cho nhiều ảnh
        """
        results = []
        for img_source, context in image_sources:
            try:
                emb = self.create_image_embedding_from_url(img_source, context)
                results.append(emb)
            except Exception as e:
                logger.warning(f"Skip image {img_source}: {e}")
                results.append([])
        return results

    # ---------- Image Index Operations ← MỚI ----------
    def upsert_image_vector(self, datapoint_id: str, vector: List[float]):
        if not self.image_index:
            raise ValueError("IMAGE_INDEX_ID is required to upsert image vectors.")
        return self.image_index.upsert_datapoints(datapoints=[{
            "datapoint_id": datapoint_id,
            "feature_vector": vector
        }])

    def upsert_image_vectors(self, pairs: Iterable[Tuple[str, List[float]]]):
        if not self.image_index:
            raise ValueError("IMAGE_INDEX_ID is required to upsert image vectors.")
        dps = [{"datapoint_id": pid, "feature_vector": vec} for pid, vec in pairs]
        return self.image_index.upsert_datapoints(datapoints=dps)

    def remove_image_vectors(self, ids: Iterable[str]):
        if not self.image_index:
            raise ValueError("IMAGE_INDEX_ID is required to remove image vectors.")
        return self.image_index.remove_datapoints(datapoint_ids=list(ids))

    def find_image_neighbors(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
        if not self.image_deployed_index_id:
            raise ValueError("IMAGE_DEPLOYED_INDEX_ID is required to query image neighbors.")

        resp = self.image_index_endpoint.find_neighbors(
            deployed_index_id=self.image_deployed_index_id,
            queries=[query_vector],
            num_neighbors=k
        )

        results: List[Tuple[str, float]] = []
        if not resp:
            return results

        first = resp[0]
        neighbors = getattr(first, "neighbors", None) or getattr(first, "nearest_neighbors", None)
        if neighbors:
            for n in neighbors:
                nid = (getattr(n, "id", None)
                       or getattr(n, "datapoint", None)
                       or getattr(n, "datapoint_id", None))
                dist = float(getattr(n, "distance", 0.0))
                if nid is not None:
                    results.append((nid, dist))
            return results

        if isinstance(first, list):
            for n in first:
                nid = getattr(n, "id", None) or getattr(n, "datapoint_id", None)
                dist = float(getattr(n, "distance", 0.0))
                if nid is not None:
                    results.append((nid, dist))
        return results

    def search_by_image(self, image_source, k: int = 5, contextual_text: str = None) -> List[Tuple[str, float]]:
        """
        Tìm kiếm bằng ảnh
        
        Args:
            image_source: URL hoặc bytes của ảnh
            k: Số lượng kết quả
            contextual_text: Text mô tả ảnh (optional)
        """
        if isinstance(image_source, bytes):
            q = self.create_image_embedding_from_bytes(image_source, contextual_text)
        else:
            q = self.create_image_embedding_from_url(image_source, contextual_text)
        return self.find_image_neighbors(q, k)

    # ---------- Text Index operations (giữ nguyên) ----------
    def upsert_vector(self, datapoint_id: str, vector: List[float]):
        if not self.index:
            raise ValueError("INDEX_ID is required to upsert vectors.")
        return self.index.upsert_datapoints(datapoints=[{
            "datapoint_id": datapoint_id,
            "feature_vector": vector
        }])

    def upsert_vectors(self, pairs: Iterable[Tuple[str, List[float]]]):
        if not self.index:
            raise ValueError("INDEX_ID is required to upsert vectors.")
        dps = [{"datapoint_id": pid, "feature_vector": vec} for pid, vec in pairs]
        return self.index.upsert_datapoints(datapoints=dps)

    def remove_vectors(self, ids: Iterable[str]):
        if not self.index:
            raise ValueError("INDEX_ID is required to remove vectors.")
        return self.index.remove_datapoints(datapoint_ids=list(ids))

    def find_neighbors(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
        if not self.deployed_index_id:
            raise ValueError("DEPLOYED_INDEX_ID is required to query neighbors.")

        resp = self.index_endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[query_vector],
            num_neighbors=k
        )

        results: List[Tuple[str, float]] = []
        if not resp:
            return results

        first = resp[0]
        neighbors = getattr(first, "neighbors", None) or getattr(first, "nearest_neighbors", None)
        if neighbors:
            for n in neighbors:
                nid = (getattr(n, "id", None)
                       or getattr(n, "datapoint", None)
                       or getattr(n, "datapoint_id", None))
                dist = float(getattr(n, "distance", 0.0))
                if nid is not None:
                    results.append((nid, dist))
            return results

        if isinstance(first, list):
            for n in first:
                nid = getattr(n, "id", None) or getattr(n, "datapoint_id", None)
                dist = float(getattr(n, "distance", 0.0))
                if nid is not None:
                    results.append((nid, dist))
        return results

    def search_by_text(self, query_text: str, k: int = 5) -> List[Tuple[str, float]]:
        q = self.create_embedding(query_text, task_type="RETRIEVAL_QUERY")
        return self.find_neighbors(q, k)
