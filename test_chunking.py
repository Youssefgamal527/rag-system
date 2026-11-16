from src.pdf_loader import load_pdf
from src.chunker import create_chunks


PDF_PATH = "knowledge/documents/guide.pdf"


def main():
    pages = load_pdf(PDF_PATH)

    chunks = create_chunks(
        pages=pages,
        source="guide.pdf"
    )

    print(f"\nTotal pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")

    print("\n" + "=" * 70)

    for chunk in chunks[:12]:
        print(f"\nChunk ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['metadata']['page']}")
        print(f"Length: {len(chunk['text'])}")
        print("-" * 70)
        print(chunk["text"])
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()