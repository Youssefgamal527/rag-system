from src.pdf_loader import load_pdf

PDF_PATH = "knowledge/documents/guide.pdf"

pages = load_pdf(PDF_PATH)

for page in pages[:3]:
    print(f"\n--- Page {page['page']} ---")
    print(page["text"][:500])
