from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings.embedding_service import EmbeddingService
from src.vector_store.qdrant_store import QdrantStore
from src.ingestion_pipeline import IngestionPipeline


# 1. Create Mock Embedding Provider
class MockEmbeddingProvider(BaseEmbeddingProvider):

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]


# 2. Initialize Embedding Service
provider = MockEmbeddingProvider()
embedding_service = EmbeddingService(provider)


# 3. Initialize Test Vector Store
vector_store = QdrantStore(
    collection_name="rag_test",
    vector_size=3
)

vector_store.create_collection()


# 4. Initialize Ingestion Pipeline
pipeline = IngestionPipeline(
    embedding_service=embedding_service,
    vector_store=vector_store
)


# 5. Ingest PDF
result = pipeline.ingest_pdf(
    "knowledge/documents/guide.pdf"
)

print("\n--- Ingestion Results ---")
print(result)


# 6. Check Stored Points
count = vector_store.client.count(
    collection_name="rag_test",
    exact=True
)

print("\nStored Points:", count.count)


# 7. Close Connection
vector_store.client.close()