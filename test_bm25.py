import pytest

from src.retrieval.bm25_retriever import BM25Retriever


@pytest.fixture
def retriever():
    documents = [
        {
            "chunk_id": "doc_1",
            "text": "Scholarships and required documents for studying abroad",
            "metadata": {"page": 1}
        },
        {
            "chunk_id": "doc_2",
            "text": "Scholarships and financial support for students",
            "metadata": {"page": 2}
        },
        {
            "chunk_id": "doc_3",
            "text": "Visa application and travel requirements",
            "metadata": {"page": 3}
        },
        {
            "chunk_id": "doc_4",
            "text": "Accommodation and student housing",
            "metadata": {"page": 4}
        },
        {
            "chunk_id": "doc_5",
            "text": "Computer science programming and software development",
            "metadata": {"page": 5}
        },
        {
            "chunk_id": "doc_6",
            "text": "University admission and academic requirements",
            "metadata": {"page": 6}
        }
    ]

    instance = BM25Retriever()
    instance.index_documents(documents)

    return instance


def test_tokenize_supports_arabic_and_english():
    tokens = BM25Retriever.tokenize(
        "Scholarships المنح الدراسية"
    )

    assert "scholarships" in tokens
    assert "المنح" in tokens
    assert "الدراسية" in tokens


def test_index_documents_rejects_empty_list():
    retriever = BM25Retriever()

    with pytest.raises(ValueError):
        retriever.index_documents([])


def test_search_rejects_uninitialized_index():
    retriever = BM25Retriever()

    with pytest.raises(ValueError):
        retriever.search("scholarships")


def test_search_rejects_empty_query(retriever):
    with pytest.raises(ValueError):
        retriever.search("   ")


def test_search_rejects_invalid_limit(retriever):
    with pytest.raises(ValueError):
        retriever.search("scholarships", limit=0)


def test_search_returns_only_matching_documents(retriever):
    results = retriever.search("scholarships")

    assert len(results) == 2

    for result in results:
        assert result["score"] > 0


def test_search_ranks_results_by_score(retriever):
    results = retriever.search("scholarships documents")

    assert len(results) >= 1
    assert results[0]["score"] > 0

    for previous, current in zip(results, results[1:]):
        assert previous["score"] >= current["score"]


def test_search_respects_limit(retriever):
    results = retriever.search(
        "scholarships",
        limit=1
    )

    assert len(results) == 1


def test_search_preserves_document_metadata(retriever):
    results = retriever.search("scholarships")

    assert "metadata" in results[0]
    assert "page" in results[0]["metadata"]
    assert "score" in results[0]