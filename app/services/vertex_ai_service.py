# services/vertex_ai_service.py
import logging, time, random
from typing import List, Tuple, Iterable, Optional
from google.api_core.exceptions import ResourceExhausted

from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

logger = logging.getLogger(__name__)

class VertexAIService:
    def __init__(
        self,
        project_id: str,
        location: str,
        index_endpoint_id: str,
        deployed_index_id: Optional[str] = None,
        index_id: Optional[str] = None,
        model_name: str = "gemini-embedding-001",   # hoặc đọc từ ENV
    ):
        aiplatform.init(project=project_id, location=location)
        vertexai.init(project=project_id, location=location)

        self.embedding_model = TextEmbeddingModel.from_pretrained(model_name)
        self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=index_endpoint_id)
        self.deployed_index_id = deployed_index_id
        self.index = aiplatform.MatchingEngineIndex(index_name=index_id) if index_id else None

        logger.info("✓ VertexAIService initialized (project=%s, location=%s, model=%s)",
                    project_id, location, model_name)

    # ---------- Retry helper ----------
    def _retry(self, fn, *, max_retries=6, base_sleep=0.6):
        for i in range(max_retries):
            try:
                return fn()
            except ResourceExhausted as e:  # 429 quota
                sleep = base_sleep * (2 ** i) + random.uniform(0, 0.25)
                logger.warning("429 quota; retry %d in %.2fs: %s", i+1, sleep, e)
                time.sleep(sleep)
        # lần cuối vẫn fail → ném cho caller
        return fn()

    # ---------- Embeddings ----------
    def create_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        inp = TextEmbeddingInput(text=text, task_type=task_type)
        return self._retry(lambda: self.embedding_model.get_embeddings([inp])[0].values)

    def create_embeddings_batch(
        self, texts: Iterable[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[List[float]]:
        inputs = [TextEmbeddingInput(text=t, task_type=task_type) for t in texts]
        return self._retry(lambda: [e.values for e in self.embedding_model.get_embeddings(inputs)])

    # ---------- Index upserts ----------
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

    # ---------- Query ----------
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
