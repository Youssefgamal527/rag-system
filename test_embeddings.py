import pytest

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


@pytest.fixture
def embedding_service():
    provider = MockEmbeddingProvider()
    return EmbeddingService(provider)


def test_generate_embedding_returns_vector(embedding_service):
    vector = embedding_service.generate_embedding("الدراسة في الخارج")

    assert isinstance(vector, list)
    assert len(vector) == 3


def test_generate_embedding_returns_float_values(embedding_service):
    vector = embedding_service.generate_embedding("Scholarships")

    assert all(isinstance(value, float) for value in vector)


def test_generate_embeddings_returns_correct_number(embedding_service):
    texts = [
        "الدراسة في الخارج",
        "المنح الدراسية",
        "الأوراق المطلوبة للسفر"
    ]

    vectors = embedding_service.generate_embeddings(texts)

    assert len(vectors) == len(texts)


def test_generate_embeddings_returns_empty_list(embedding_service):
    assert embedding_service.generate_embeddings([]) == []


def test_generate_embedding_rejects_empty_text(embedding_service):
    with pytest.raises(ValueError):
        embedding_service.generate_embedding("   ")


def test_generate_embeddings_rejects_empty_text(embedding_service):
    with pytest.raises(ValueError):
        embedding_service.generate_embeddings([
            "الدراسة في الخارج",
            "   "
        ])