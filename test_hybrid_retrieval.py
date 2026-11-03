
from src.embeddings.ollama_provider import OllamaEmbeddingProvider
from src.embeddings.embedding_service import EmbeddingService

from src.vector_store.qdrant_store import QdrantStore

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.reranker import RRFFusion
from src.retrieval.hybrid_retriever import HybridRetriever


# 1. Prepare components
provider = OllamaEmbeddingProvider()
embedding_service = EmbeddingService(provider)

store = QdrantStore()

# 2. Test documents
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

# 3. Create embeddings and store documents
store.create_collection()

embeddings = embedding_service.generate_embeddings(
    [chunk["text"] for chunk in chunks]
)

store.add_chunks(chunks, embeddings)

# 4. Initialize BM25
bm25 = BM25Retriever()
bm25.index_documents(chunks)

# 5. Initialize Dense Retriever
dense = DenseRetriever(
    embedding_service=embedding_service,
    vector_store=store
)

# 6. Initialize Hybrid Retriever
rrf = RRFFusion(k=60)

hybrid = HybridRetriever(
    bm25_retriever=bm25,
    dense_retriever=dense,
    rrf=rrf
)

# 7. Search
query = "ما هي شروط القبول في الجامعة؟"

print("\nBM25 RESULTS")
print("=" * 50)

for result in bm25.search(query, limit=3):
    print(result["chunk_id"], result["score"])
    print(result["text"])

print("\nDENSE RESULTS")
print("=" * 50)

for result in dense.search(query, limit=3):
    print(result["chunk_id"], result["score"])
    print(result["text"])

print("\nHYBRID RESULTS")
print("=" * 50)

for result in hybrid.search(
    query=query,
    limit=3,
    retrieval_limit=3
):
    print(result["chunk_id"], result["rrf_score"])
    print(result["text"])
    print("-" * 50)
