
from pathlib import Path
import pymupdf

from src.text_cleaner import clean_text
from src.pdf_text_reconstructor import reconstruct_page_text


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract and reconstruct PDF text while preserving page numbers.
    """

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    pages = []

    with pymupdf.open(pdf_path) as document:

        for page_number, page in enumerate(document, start=1):

            text = reconstruct_page_text(page)
            text = clean_text(text)

            pages.append({
                "page": page_number,
                "text": text
            })

    return pages
