import uuid
import pytest

from reportlab.pdfgen import canvas

from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings.embedding_service import EmbeddingService
from src.vector_store.qdrant_store import QdrantStore
from src.ingestion_pipeline import IngestionPipeline


# Mock Embedding Provider
class MockEmbeddingProvider(BaseEmbeddingProvider):

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]


@pytest.fixture
def pipeline():

    collection_name = f"rag_ingestion_test_{uuid.uuid4().hex[:8]}"

    provider = MockEmbeddingProvider()
    embedding_service = EmbeddingService(provider)

    vector_store = QdrantStore(
        collection_name=collection_name,
        vector_size=3
    )

    vector_store.create_collection()

    ingestion_pipeline = IngestionPipeline(
        embedding_service=embedding_service,
        vector_store=vector_store
    )

    yield ingestion_pipeline, vector_store

    vector_store.client.delete_collection(
        collection_name=collection_name
    )

    vector_store.client.close()


@pytest.fixture
def sample_pdf(tmp_path):

    pdf_path = tmp_path / "sample.pdf"

    pdf = canvas.Canvas(str(pdf_path))

    pdf.drawString(
        100,
        750,
        "Scholarships help students study abroad."
    )

    pdf.drawString(
        100,
        720,
        "Students must prepare the required documents."
    )

    pdf.save()

    return str(pdf_path)


@pytest.fixture
def empty_pdf(tmp_path):

    pdf_path = tmp_path / "empty.pdf"

    pdf = canvas.Canvas(str(pdf_path))
    pdf.save()

    return str(pdf_path)


def test_ingest_pdf_returns_success(
    pipeline,
    sample_pdf
):

    ingestion_pipeline, _ = pipeline

    result = ingestion_pipeline.ingest_pdf(sample_pdf)

    assert result["status"] == "Success"
    assert result["source"] == "sample.pdf"
    assert result["pages"] == 1
    assert result["chunks"] > 0


def test_ingest_pdf_stores_chunks(
    pipeline,
    sample_pdf
):

    ingestion_pipeline, vector_store = pipeline

    result = ingestion_pipeline.ingest_pdf(sample_pdf)

    count = vector_store.client.count(
        collection_name=vector_store.collection_name,
        exact=True
    )

    assert count.count == result["chunks"]


def test_ingest_empty_pdf(
    pipeline,
    empty_pdf
):

    ingestion_pipeline, vector_store = pipeline

    result = ingestion_pipeline.ingest_pdf(empty_pdf)

    assert result["status"] == "No text found"
    assert result["chunks"] == 0

    count = vector_store.client.count(
        collection_name=vector_store.collection_name,
        exact=True
    )

    assert count.count == 0


def test_ingest_pdf_file_not_found(pipeline):

    ingestion_pipeline, _ = pipeline

    with pytest.raises(FileNotFoundError):
        ingestion_pipeline.ingest_pdf("nonexistent.pdf")