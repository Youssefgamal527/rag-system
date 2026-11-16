
from src.vector_store.qdrant_store import QdrantStore

store = QdrantStore()

chunks = store.get_all_chunks()

print("Total chunks:", len(chunks))

for chunk in chunks[:3]:
    print("\nChunk ID:", chunk["chunk_id"])
    print("Source:", chunk["metadata"]["source"])
    print("Page:", chunk["metadata"]["page"])
    print("Text:", chunk["text"][:150])
