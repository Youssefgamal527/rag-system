
from src.vector_store.qdrant_store import QdrantStore
from src.retrieval.bm25_retriever import BM25Retriever


# 1. Load chunks from Qdrant
vector_store = QdrantStore()

chunks = vector_store.get_all_chunks()

print("Loaded chunks:", len(chunks))


# 2. Build BM25 index
bm25 = BM25Retriever()

bm25.index_documents(chunks)

print("BM25 index initialized successfully!")


# 3. Test keyword search
query = "شروط القبول في الجامعات"

results = bm25.search(
    query=query,
    limit=3
)


# 4. Display results
print("\nSearch Results:")

for result in results:

    print("\nChunk ID:", result["chunk_id"])
    print("Score:", result["score"])
    print("Text:", result["text"])
