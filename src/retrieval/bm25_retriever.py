import re

from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self):

        self.documents = []
        self.bm25 = None

    @staticmethod
    def tokenize(text: str) -> list[str]:

        text = text.lower()

        tokens = re.findall(
            r"[\u0600-\u06FF]+|[a-z0-9]+",
            text
        )

        return tokens

    def index_documents(self, chunks: list[dict]) -> None:

        if not chunks:
            raise ValueError("Chunks cannot be empty")

        self.documents = chunks

        tokenized_documents = [
            self.tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(
        self,
        query: str,
        limit: int = 5
    ) -> list[dict]:

        if self.bm25 is None:
            raise ValueError("BM25 index has not been initialized")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if limit <= 0:
            raise ValueError("Limit must be greater than zero")

        tokenized_query = self.tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)
        
        # Keep only documents with positive scores
        matched_indices = [
            index
            for index, score in enumerate(scores)
            if score > 0
        ]

        ranked_indices = sorted(
            matched_indices,
            key=lambda index: scores[index],
            reverse=True
        )[:limit]

        results = []

        for index in ranked_indices:

            chunk = self.documents[index]

            results.append({
                **chunk,
                "score": float(scores[index])
            })

        return results