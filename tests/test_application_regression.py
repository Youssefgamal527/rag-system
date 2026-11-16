
from unittest.mock import Mock

import pytest

from src.application import RAGApplication


@pytest.fixture
def app():
    application = RAGApplication()

    application.vector_store = Mock()
    application.bm25_retriever = Mock()
    application.answer_generator = Mock()
    application.ingestion_pipeline = Mock()

    return application


def test_ask_rejects_uninitialized_application(app):
    with pytest.raises(
        RuntimeError,
        match="Application must be initialized first"
    ):
        app.ask("What is RAG?")


def test_ask_rejects_empty_query(app):
    app.initialized = True

    with pytest.raises(
        ValueError,
        match="Query cannot be empty"
    ):
        app.ask("")


def test_ask_handles_empty_knowledge_base(app):
    app.initialized = True
    app.documents_count = 0

    result = app.ask("What is RAG?")

    assert result["answer"] == (
        "No documents have been added to the knowledge base yet."
    )
    assert result["sources"] == []
    assert result["retrieved_chunks"] == []

    app.answer_generator.generate.assert_not_called()


def test_ask_calls_answer_generator(app):
    app.initialized = True
    app.documents_count = 10

    expected_result = {
        "answer": "RAG combines retrieval and generation.",
        "sources": [
            {"source": "guide.pdf", "page": 5}
        ],
        "retrieved_chunks": []
    }

    app.answer_generator.generate.return_value = expected_result

    result = app.ask("What is RAG?")

    app.answer_generator.generate.assert_called_once_with(
        query="What is RAG?"
    )

    assert result == expected_result


def test_initialize_loads_chunks_and_builds_bm25_index(app):
    chunks = [
        {"chunk_id": "chunk_001", "text": "First chunk"},
        {"chunk_id": "chunk_002", "text": "Second chunk"}
    ]

    app.vector_store.get_all_chunks.return_value = chunks

    result = app.initialize()

    app.vector_store.create_collection.assert_called_once()
    app.vector_store.get_all_chunks.assert_called_once()

    app.bm25_retriever.index_documents.assert_called_once_with(chunks)

    assert result == {
        "status": "Success",
        "chunks_loaded": 2
    }

    assert app.initialized is True
    assert app.documents_count == 2
