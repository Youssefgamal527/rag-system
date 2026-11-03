from src.embeddings.embedding_service import EmbeddingService
from src.vector_store.qdrant_store import QdrantStore


class DenseRetriever:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantStore
    ):

        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        limit: int = 5
    ) -> list[dict]:

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if limit <= 0:
            raise ValueError("Limit must be greater than zero")

        # 1. Convert query into embedding
        query_embedding = self.embedding_service.generate_embedding(
            query
        )

        # 2. Search in Qdrant
        results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=limit
        )

        return results