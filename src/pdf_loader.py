from pathlib import Path
from pypdf import PdfReader
from src.text_cleaner import clean_text


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF file while preserving page numbers.
    """

    pdf_path = Path(file_path)

    # Check if the file exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    # Read the PDF
    reader = PdfReader(pdf_path)

    pages = []

    # Extract text from each page
    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""
        text = clean_text(text)

        pages.append({
            "page": page_number,
            "text": text
        })

    return pages