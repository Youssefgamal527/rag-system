from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings.embedding_service import EmbeddingService


class MockEmbeddingProvider(BaseEmbeddingProvider):

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]


provider = MockEmbeddingProvider()

embedding_service = EmbeddingService(provider)


# Test a single text
vector = embedding_service.generate_embedding(
    "الدراسة في الخارج"
)

print("Single Embedding:", vector)
print("Vector Dimensions:", len(vector))


# Test multiple texts
vectors = embedding_service.generate_embeddings([
    "الدراسة في الخارج",
    "المنح الدراسية",
    "الأوراق المطلوبة للسفر"
])

print("\nBatch Embeddings:", vectors)
print("Number of Vectors:", len(vectors))


# Test empty input
try:
    embedding_service.generate_embedding("   ")
except ValueError as error:
    print("\nEmpty Text Test:", error)