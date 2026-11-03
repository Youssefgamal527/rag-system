import pytest

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import RRFFusion


class MockBM25Retriever:

    def search(self, query: str, limit: int = 5) -> list[dict]:

        results = [
            {"chunk_id": "A", "text": "Document A"},
            {"chunk_id": "B", "text": "Document B"},
            {"chunk_id": "C", "text": "Document C"}
        ]

        return results[:limit]


class MockDenseRetriever:

    def search(self, query: str, limit: int = 5) -> list[dict]:

        results = [
            {"chunk_id": "B", "text": "Document B"},
            {"chunk_id": "D", "text": "Document D"},
            {"chunk_id": "A", "text": "Document A"}
        ]

        return results[:limit]


@pytest.fixture
def hybrid_retriever():

    return HybridRetriever(
        bm25_retriever=MockBM25Retriever(),
        dense_retriever=MockDenseRetriever(),
        rrf=RRFFusion(k=60)
    )


def test_hybrid_search_returns_results(hybrid_retriever):

    results = hybrid_retriever.search(
        query="Scholarships",
        limit=5
    )

    assert isinstance(results, list)
    assert len(results) == 4


def test_hybrid_search_merges_results(hybrid_retriever):

    results = hybrid_retriever.search(
        query="Scholarships"
    )

    chunk_ids = {
        result["chunk_id"]
        for result in results
    }

    assert chunk_ids == {"A", "B", "C", "D"}


def test_hybrid_search_ranks_common_documents_first(
    hybrid_retriever
):

    results = hybrid_retriever.search(
        query="Scholarships"
    )

    assert results[0]["chunk_id"] == "B"
    assert results[1]["chunk_id"] == "A"


def test_hybrid_search_returns_rrf_scores(hybrid_retriever):

    results = hybrid_retriever.search(
        query="Scholarships"
    )

    for result in results:
        assert "rrf_score" in result
        assert result["rrf_score"] > 0


def test_hybrid_search_respects_limit(hybrid_retriever):

    results = hybrid_retriever.search(
        query="Scholarships",
        limit=2
    )

    assert len(results) == 2


def test_hybrid_search_passes_retrieval_limit(hybrid_retriever):

    results = hybrid_retriever.search(
        query="Scholarships",
        limit=2,
        retrieval_limit=1
    )

    assert len(results) == 2


def test_hybrid_search_rejects_empty_query(hybrid_retriever):

    with pytest.raises(ValueError):
        hybrid_retriever.search(query="   ")


@pytest.mark.parametrize(
    "limit,retrieval_limit",
    [
        (0, 10),
        (-1, 10),
        (5, 0),
        (5, -1)
    ]
)
def test_hybrid_search_rejects_invalid_limits(
    hybrid_retriever,
    limit,
    retrieval_limit
):

    with pytest.raises(ValueError):
        hybrid_retriever.search(
            query="Scholarships",
            limit=limit,
            retrieval_limit=retrieval_limit
        )