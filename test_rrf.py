from src.retrieval.reranker import RRFFusion


# 1. Create sample BM25 results
bm25_results = [
    {"chunk_id": "A", "text": "Document A"},
    {"chunk_id": "B", "text": "Document B"},
    {"chunk_id": "C", "text": "Document C"}
]


# 2. Create sample Dense Search results
dense_results = [
    {"chunk_id": "B", "text": "Document B"},
    {"chunk_id": "D", "text": "Document D"},
    {"chunk_id": "A", "text": "Document A"}
]


# 3. Initialize RRF
rrf = RRFFusion(k=60)


# 4. Fuse results
final_results = rrf.fuse(
    result_lists=[
        bm25_results,
        dense_results
    ],
    limit=5
)


# 5. Display results
print("\n--- RRF Fusion Results ---")

for result in final_results:

    print(
        f"Chunk: {result['chunk_id']} | "
        f"Score: {result['rrf_score']:.6f}"
    )