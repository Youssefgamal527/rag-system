
from src.application import RAGApplication

app = RAGApplication()
app.initialize()

query = "ما الموضوعات الرئيسية التي يتناولها هذا الدليل؟"

retrievers = {
    "BM25": app.bm25_retriever,
    "Dense": app.dense_retriever,
    "Hybrid": app.hybrid_retriever
}

for name, retriever in retrievers.items():

    print("\n" + "=" * 60)
    print(f"{name} RESULTS")
    print("=" * 60)

    if name == "Hybrid":
        results = retriever.search(
            query=query,
            limit=5,
            retrieval_limit=10
        )
    else:
        results = retriever.search(
            query=query,
            limit=5
        )

    for i, result in enumerate(results, 1):

        metadata = result.get("metadata", {})

        print(f"\nResult {i}")
        print("Page:", metadata.get("page"))
        print("Chunk:", result["chunk_id"])
        print("Score:", result.get("score", result.get("rrf_score")))
        print("Text:", result["text"][:500])
