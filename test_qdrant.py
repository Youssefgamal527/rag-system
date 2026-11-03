from src.vector_store.qdrant_store import QdrantStore


# Initialize Qdrant Store
store = QdrantStore()

# Create collection if it doesn't exist
store.create_collection()


# Create sample chunks
chunks = [
    {
        "chunk_id": "test_p1_c1",
        "text": "الدراسة في الخارج تتطلب تجهيز الأوراق المطلوبة.",
        "metadata": {
            "source": "test.pdf",
            "page": 1,
            "chunk_index": 1
        }
    },
    {
        "chunk_id": "test_p1_c2",
        "text": "المنح الدراسية تساعد الطلاب على استكمال تعليمهم.",
        "metadata": {
            "source": "test.pdf",
            "page": 1,
            "chunk_index": 2
        }
    }
]


# Create mock embeddings
vector_1 = [0.0] * 1024
vector_1[0] = 1.0

vector_2 = [0.0] * 1024
vector_2[1] = 1.0

embeddings = [vector_1, vector_2]


# Test inserting data
store.add_chunks(chunks, embeddings)

print("Chunks inserted successfully!")


# Test searching
results = store.search(
    query_embedding=vector_1,
    limit=2
)


print("\n--- Search Results ---")

for result in results:

    print("\nChunk ID:", result["chunk_id"])
    print("Text:", result["text"])
    print("Metadata:", result["metadata"])
    print("Score:", result["score"])


# Close connection
store.client.close()