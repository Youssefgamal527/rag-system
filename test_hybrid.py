from src.pdf_loader import load_pdf
from src.chunker import create_chunks

from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings.embedding_service import EmbeddingService

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.reranker import RRFFusion
from src.retrieval.hybrid_retriever import HybridRetriever


# 1. Mock Embedding Provider
class MockEmbeddingProvider(BaseEmbeddingProvider):

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]


# 2. Mock Vector Store
class MockVectorStore:

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5
    ) -> list[dict]:

        return [
            {
                **chunk,
                "score": 0.9
            }
            for chunk in self.chunks[5:5 + limit]
        ]


# 3. Load PDF and Create Chunks
pages = load_pdf("knowledge/documents/guide.pdf")

chunks = create_chunks(
    pages,
    source="guide.pdf"
)


# 4. Initialize BM25
bm25_retriever = BM25Retriever()
bm25_retriever.index_documents(chunks)


# 5. Initialize Dense Retriever
embedding_service = EmbeddingService(
    MockEmbeddingProvider()
)

mock_vector_store = MockVectorStore(chunks)

dense_retriever = DenseRetriever(
    embedding_service=embedding_service,
    vector_store=mock_vector_store
)


# 6. Initialize RRF
rrf = RRFFusion(k=60)


# 7. Initialize Hybrid Retriever
hybrid_retriever = HybridRetriever(
    bm25_retriever=bm25_retriever,
    dense_retriever=dense_retriever,
    rrf=rrf
)


# 8. Search
query = "المنح الدراسية والأوراق المطلوبة"

results = hybrid_retriever.search(
    query=query,
    limit=5,
    retrieval_limit=10
)


# 9. Display Results
print("\n--- Hybrid Search Results ---")

for index, result in enumerate(results, start=1):

    print(f"\nRank: {index}")
    print("Chunk ID:", result["chunk_id"])
    print("Page:", result["metadata"]["page"])
    print("RRF Score:", round(result["rrf_score"], 6))
    print("Text:", result["text"][:250])
    