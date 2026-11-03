from src.pdf_loader import load_pdf
from src.chunker import create_chunks
from src.chunker import split_text


PDF_PATH = "knowledge/documents/guide.pdf"

pages = load_pdf(PDF_PATH)

chunks = create_chunks(
    pages,
    source="guide.pdf"
)

print("\n--- Chunk Statistics ---")

chunk_lengths = [len(chunk["text"]) for chunk in chunks]

print(f"Total Pages: {len(pages)}")
print("Total Chunks:", len(chunks))
print("Average Length:", sum(chunk_lengths) / len(chunk_lengths))
print("Maximum Length:", max(chunk_lengths))
print("Minimum Length:", min(chunk_lengths))


for chunk in chunks[:3]:

    print("\n--------------------------")
    print("Chunk ID:", chunk["chunk_id"])
    print("Metadata:", chunk["metadata"])
    print("Text:", chunk["text"][:300])


# Create sample text
sample_text = " ".join(
    f"word{i}" for i in range(1000)
)

sample_chunks = split_text(
    sample_text,
    chunk_size=1800,
    chunk_overlap=300
)

print("\n--- Overlap Test ---")

print("Sample Chunks:", len(sample_chunks))

for i in range(len(sample_chunks) - 1):

    previous = sample_chunks[i]
    current = sample_chunks[i + 1]

    overlap = 0

    for size in range(min(len(previous), len(current)), 0, -1):

        if previous[-size:] == current[:size]:
            overlap = size
            break

    print(f"Chunks {i+1} & {i+2}: Overlap = {overlap} characters")