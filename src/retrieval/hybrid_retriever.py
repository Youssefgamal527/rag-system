from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.reranker import RRFFusion


class HybridRetriever:

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf: RRFFusion
    ):

        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf = rrf

    def search(
        self,
        query: str,
        limit: int = 5,
        retrieval_limit: int = 10
    ) -> list[dict]:

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if limit <= 0 or retrieval_limit <= 0:
            raise ValueError("Limits must be greater than zero")

        # 1. Retrieve results using BM25
        bm25_results = self.bm25_retriever.search(
            query=query,
            limit=retrieval_limit
        )

        # 2. Retrieve results using Dense Search
        dense_results = self.dense_retriever.search(
            query=query,
            limit=retrieval_limit
        )

        # 3. Merge results using RRF
        final_results = self.rrf.fuse(
            result_lists=[
                bm25_results,
                dense_results
            ],
            limit=limit
        )

        return final_results