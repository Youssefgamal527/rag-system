import pytest

from src.retrieval.reranker import RRFFusion


@pytest.fixture
def rrf():
    return RRFFusion(k=60)


@pytest.fixture
def sample_results():
    bm25_results = [
        {"chunk_id": "A", "text": "Document A"},
        {"chunk_id": "B", "text": "Document B"},
        {"chunk_id": "C", "text": "Document C"}
    ]

    dense_results = [
        {"chunk_id": "B", "text": "Document B"},
        {"chunk_id": "D", "text": "Document D"},
        {"chunk_id": "A", "text": "Document A"}
    ]

    return [bm25_results, dense_results]


def test_rrf_returns_fused_results(rrf, sample_results):
    results = rrf.fuse(sample_results)

    assert isinstance(results, list)
    assert len(results) == 4


def test_rrf_combines_scores_for_common_documents(
    rrf,
    sample_results
):
    results = rrf.fuse(sample_results)

    scores = {
        result["chunk_id"]: result["rrf_score"]
        for result in results
    }

    expected_score_a = (1 / 61) + (1 / 63)
    expected_score_b = (1 / 62) + (1 / 61)

    assert scores["A"] == pytest.approx(expected_score_a)
    assert scores["B"] == pytest.approx(expected_score_b)


def test_rrf_ranks_results_by_score(rrf, sample_results):
    results = rrf.fuse(sample_results)

    for previous, current in zip(results, results[1:]):
        assert previous["rrf_score"] >= current["rrf_score"]


def test_rrf_places_common_documents_first(rrf, sample_results):
    results = rrf.fuse(sample_results)

    assert results[0]["chunk_id"] == "B"
    assert results[1]["chunk_id"] == "A"


def test_rrf_respects_limit(rrf, sample_results):
    results = rrf.fuse(sample_results, limit=2)

    assert len(results) == 2


def test_rrf_handles_empty_result_lists(rrf):
    results = rrf.fuse([[], []])

    assert results == []


def test_rrf_rejects_invalid_limit(rrf):
    with pytest.raises(ValueError):
        rrf.fuse([], limit=0)


def test_rrf_rejects_invalid_k():
    with pytest.raises(ValueError):
        RRFFusion(k=0)