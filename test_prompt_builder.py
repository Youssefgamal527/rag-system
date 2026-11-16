from src.generation.prompt_builder import PromptBuilder


builder = PromptBuilder()

chunks = [
    {
        "chunk_id": "chunk_001",
        "text": "RAG combines information retrieval with text generation.",
        "metadata": {
            "source": "rag_guide.pdf",
            "page": 3
        },
        "rrf_score": 0.032
    },
    {
        "chunk_id": "chunk_002",
        "text": "Hybrid Search combines BM25 and Dense Retrieval.",
        "metadata": {
            "source": "rag_guide.pdf",
            "page": 5
        },
        "rrf_score": 0.016
    }
]

query = "What is Hybrid Search?"

prompt = builder.build(
    query=query,
    retrieved_chunks=chunks
)

print(prompt)