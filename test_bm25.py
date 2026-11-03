from src.pdf_loader import load_pdf
from src.chunker import create_chunks
from src.retrieval.bm25_retriever import BM25Retriever


# 1. Load PDF
pages = load_pdf("knowledge/documents/guide.pdf")


# 2. Create Chunks
chunks = create_chunks(
    pages,
    source="guide.pdf"
)


# 3. Initialize BM25 Retriever
retriever = BM25Retriever()


# 4. Build BM25 Index
retriever.index_documents(chunks)

print("BM25 Index Created Successfully!")
print("Total Documents:", len(chunks))


# 5. Search
query = "المنح الدراسية والأوراق المطلوبة"

results = retriever.search(
    query=query,
    limit=5
)


# 6. Display Results
print("\n--- BM25 Search Results ---")

for result in results:

    print("\nChunk ID:", result["chunk_id"])
    print("Page:", result["metadata"]["page"])
    print("Score:", round(result["score"], 4))
    print("Text:", result["text"][:300])
    