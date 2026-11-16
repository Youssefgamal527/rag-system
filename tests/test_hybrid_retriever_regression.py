
import pytest

from src.retrieval.hybrid_retriever import HybridRetriever


class FakeBM25:
    def search(self, query, limit):
        return [
            {"chunk_id": "bm25_001", "text": "BM25 result"}
        ]


class FakeDense:
    def search(self, query, limit):
        return [
            {"chunk_id": "dense_001", "text": "Dense result"}
        ]


class FakeRRF:
    def fuse(self, result_lists, limit):
        assert len(result_lists) == 2
        return [
            {"chunk_id": "bm25_001", "text": "BM25 result"},
            {"chunk_id": "dense_001", "text": "Dense result"}
        ][:limit]


class FakeVectorStore:
    def search_by_text(self, text, limit):
        rows = {
            "ألمانيا": [
                {
                    "chunk_id": "germany_001",
                    "text": "الدولة: ألمانيا | معيشة شهريا: 850 – 1,100 يورو"
                }
            ],
            "إسبانيا": [
                {
                    "chunk_id": "spain_001",
                    "text": "الدولة: إسبانيا | معيشة شهريا: 900 – 1,360 يورو"
                }
            ]
        }

        return rows.get(text, [])


@pytest.fixture
def retriever():
    return HybridRetriever(
        bm25_retriever=FakeBM25(),
        dense_retriever=FakeDense(),
        rrf=FakeRRF(),
        vector_store=FakeVectorStore()
    )


def test_hybrid_search_returns_bm25_and_dense_results(retriever):
    results = retriever.search("Hybrid Search")

    ids = [item["chunk_id"] for item in results]

    assert "bm25_001" in ids
    assert "dense_001" in ids


def test_hybrid_search_retrieves_requested_country_rows(retriever):
    results = retriever.search(
        "قارن مصاريف المعيشة شهريا في ألمانيا وإسبانيا"
    )

    ids = [item["chunk_id"] for item in results]

    assert "germany_001" in ids
    assert "spain_001" in ids


def test_hybrid_search_does_not_duplicate_chunks(retriever):
    results = retriever.search("Hybrid Search")

    ids = [item["chunk_id"] for item in results]

    assert len(ids) == len(set(ids))


def test_hybrid_search_without_vector_store():
    retriever = HybridRetriever(
        bm25_retriever=FakeBM25(),
        dense_retriever=FakeDense(),
        rrf=FakeRRF()
    )

    results = retriever.search("Hybrid Search")

    ids = [item["chunk_id"] for item in results]

    assert "bm25_001" in ids
    assert "dense_001" in ids


def test_hybrid_search_rejects_empty_query(retriever):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.search("")


def test_hybrid_search_rejects_invalid_limits(retriever):
    with pytest.raises(ValueError, match="Limits must be greater than zero"):
        retriever.search("Hybrid Search", limit=0)
