
from src.embeddings.ollama_provider import OllamaEmbeddingProvider
from src.embeddings.embedding_service import EmbeddingService
from src.vector_store.qdrant_store import QdrantStore


provider = OllamaEmbeddingProvider()
embedding_service = EmbeddingService(provider)
store = QdrantStore()

chunks = [
    {
        "chunk_id": "test_001",
        "text": "شروط القبول في الجامعات تتطلب الحصول على شهادة الثانوية العامة واجتياز متطلبات التقديم.",
        "metadata": {
            "source": "test",
            "page": 1,
            "chunk_index": 0
        }
    },
    {
        "chunk_id": "test_002",
        "text": "تقدم الجامعات منحًا دراسية للطلاب الدوليين وفقًا للتفوق الأكاديمي.",
        "metadata": {
            "source": "test",
            "page": 2,
            "chunk_index": 1
        }
    },
    {
        "chunk_id": "test_003",
        "text": "تتطلب إجراءات استخراج جواز السفر تقديم بطاقة الرقم القومي وصور شخصية.",
        "metadata": {
            "source": "test",
            "page": 3,
            "chunk_index": 2
        }
    }
]

# Generate embeddings
texts = [chunk["text"] for chunk in chunks]
embeddings = embedding_service.generate_embeddings(texts)

# Store chunks
store.add_chunks(chunks, embeddings)

print("Chunks stored successfully!")

# Search
query = "ما هي شروط القبول في الجامعة؟"
query_embedding = embedding_service.generate_embedding(query)

results = store.search(query_embedding, limit=3)

print("\nSearch results:")

for result in results:
    print("Score:", round(result["score"], 4))
    print("Text:", result["text"])
    print("Metadata:", result["metadata"])
    print("-" * 50)
