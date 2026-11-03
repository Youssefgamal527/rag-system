import pytest

from src.pdf_loader import load_pdf

PDF_PATH = "knowledge/documents/guide.pdf"


def test_load_pdf_returns_pages():
    pages = load_pdf(PDF_PATH)

    assert isinstance(pages, list)
    assert len(pages) > 0


def test_load_pdf_preserves_page_numbers():
    pages = load_pdf(PDF_PATH)

    assert pages[0]["page"] == 1


def test_load_pdf_returns_text():
    pages = load_pdf(PDF_PATH)

    for page in pages:
        assert "page" in page
        assert "text" in page
        assert isinstance(page["text"], str)


def test_load_pdf_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_pdf("nonexistent.pdf")
