import pytest

from src.chunker import split_text, create_chunks


def test_split_text_returns_empty_list_for_empty_text():
    assert split_text("") == []
    assert split_text("   ") == []


def test_split_text_respects_chunk_size():
    text = " ".join(f"word{i}" for i in range(500))

    chunks = split_text(
        text,
        chunk_size=100,
        chunk_overlap=20
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_split_text_preserves_overlap():
    text = " ".join(f"word{i}" for i in range(100))

    chunks = split_text(
        text,
        chunk_size=100,
        chunk_overlap=20
    )

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:]):
        previous_words = previous.split()
        current_words = current.split()

        assert previous_words[-2:] == current_words[:2]


def test_split_text_rejects_invalid_chunk_size():
    with pytest.raises(ValueError):
        split_text("Hello World", chunk_size=0)


def test_split_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("Hello World", chunk_size=100, chunk_overlap=100)


def test_create_chunks_returns_expected_structure():
    pages = [
        {"page": 1, "text": "First page content " * 20},
        {"page": 2, "text": "Second page content " * 20}
    ]

    chunks = create_chunks(pages, source="guide.pdf")

    assert len(chunks) > 0

    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "metadata" in chunk


def test_create_chunks_preserves_metadata():
    pages = [
        {"page": 3, "text": "Sample content " * 20}
    ]

    chunks = create_chunks(pages, source="guide.pdf")

    assert chunks[0]["chunk_id"] == "guide.pdf_p3_c1"
    assert chunks[0]["metadata"]["source"] == "guide.pdf"
    assert chunks[0]["metadata"]["page"] == 3
    assert chunks[0]["metadata"]["chunk_index"] == 1
