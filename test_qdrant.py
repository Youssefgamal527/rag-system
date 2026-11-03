import uuid

import pytest

from src.vector_store.qdrant_store import QdrantStore


@pytest.fixture
def store():
    # Create a unique collection for each test
    collection_name = f"rag_test_{uuid.uuid4().hex[:8]}"

    instance = QdrantStore(
        collection_name=collection_name,
        vector_size=1024
    )

    instance.create_collection()

    yield instance

    # Delete only the collection created by this test
    instance.client.delete_collection(
        collection_name=collection_name
    )

    instance.client.close()


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "test_p1_c1",
            "text": "الدراسة في الخارج تتطلب تجهيز الأوراق المطلوبة.",
            "metadata": {
                "source": "test.pdf",
                "page": 1,
                "chunk_index": 1
            }
        },
        {
            "chunk_id": "test_p1_c2",
            "text": "المنح الدراسية تساعد الطلاب على استكمال تعليمهم.",
            "metadata": {
                "source": "test.pdf",
                "page": 1,
                "chunk_index": 2
            }
        }
    ]


@pytest.fixture
def sample_embeddings():
    vector_1 = [0.0] * 1024
    vector_1[0] = 1.0

    vector_2 = [0.0] * 1024
    vector_2[1] = 1.0

    return [vector_1, vector_2]


def test_create_collection(store):
    collections = store.client.get_collections().collections

    names = [collection.name for collection in collections]

    assert store.collection_name in names


def test_create_collection_is_idempotent(store):
    # Calling create_collection again should not fail
    store.create_collection()

    collections = store.client.get_collections().collections
    names = [collection.name for collection in collections]

    assert names.count(store.collection_name) == 1


def test_add_chunks_and_search(
    store,
    sample_chunks,
    sample_embeddings
):
    store.add_chunks(sample_chunks, sample_embeddings)

    results = store.search(
        query_embedding=sample_embeddings[0],
        limit=2
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "test_p1_c1"


def test_search_preserves_chunk_data(
    store,
    sample_chunks,
    sample_embeddings
):
    store.add_chunks(sample_chunks, sample_embeddings)

    results = store.search(
        query_embedding=sample_embeddings[0],
        limit=2
    )

    result = results[0]

    assert result["text"] == sample_chunks[0]["text"]
    assert result["metadata"]["source"] == "test.pdf"
    assert result["metadata"]["page"] == 1
    assert result["metadata"]["chunk_index"] == 1
    assert isinstance(result["score"], float)


def test_add_chunks_rejects_mismatched_lengths(
    store,
    sample_chunks,
    sample_embeddings
):
    with pytest.raises(ValueError):
        store.add_chunks(
            sample_chunks,
            sample_embeddings[:1]
        )


def test_add_chunks_handles_empty_list(store):
    result = store.add_chunks([], [])

    assert result is None


def test_add_chunks_rejects_invalid_vector_size(
    store,
    sample_chunks
):
    invalid_embeddings = [[0.1, 0.2, 0.3]] * 2

    with pytest.raises(ValueError):
        store.add_chunks(
            sample_chunks,
            invalid_embeddings
        )


def test_search_rejects_invalid_vector_size(store):
    with pytest.raises(ValueError):
        store.search(
            query_embedding=[0.1, 0.2, 0.3]
        )


def test_search_rejects_invalid_limit(
    store,
    sample_embeddings
):
    with pytest.raises(ValueError):
        store.search(
            query_embedding=sample_embeddings[0],
            limit=0
        )