from pathlib import Path

from src.pdf_loader import load_pdf
from src.chunker import create_chunks
from src.embeddings.embedding_service import EmbeddingService
from src.vector_store.qdrant_store import QdrantStore


class IngestionPipeline:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantStore
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def ingest_pdf(self, file_path: str) -> dict:

        # 1. Extract text from PDF
        pages = load_pdf(file_path)

        # 2. Split extracted text into chunks
        source = Path(file_path).name
        chunks = create_chunks(pages, source)

        if not chunks:
            return {
                "source": source,
                "pages": len(pages),
                "chunks": 0,
                "status": "No text found"
            }

        # 3. Generate embeddings
        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.embedding_service.generate_embeddings(texts)

        # 4. Store chunks and embeddings in Qdrant
        self.vector_store.add_chunks(
            chunks,
            embeddings
        )

        return {
            "source": source,
            "pages": len(pages),
            "chunks": len(chunks),
            "status": "Success"
        }