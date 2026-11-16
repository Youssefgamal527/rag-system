
import re
import unicodedata

from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self):
        self.documents = []
        self.bm25 = None

    @staticmethod
    def normalize_arabic(text: str) -> str:

        text = unicodedata.normalize("NFKC", text)
        text = text.lower()

        # Remove Arabic diacritics
        text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

        # Remove Tatweel
        text = text.replace("ـ", "")

        # Normalize Arabic letters
        text = re.sub(r"[إأآٱ]", "ا", text)
        text = text.replace("ى", "ي")
        text = text.replace("ؤ", "و")
        text = text.replace("ئ", "ي")

        return text

    @staticmethod
    def tokenize(text: str) -> list[str]:

        text = BM25Retriever.normalize_arabic(text)

        tokens = re.findall(
            r"[\u0600-\u06FF]+|[a-z0-9]+",
            text
        )

        return tokens

    def index_documents(self, chunks: list[dict]) -> None:

        if not chunks:
            self.documents = []
            self.bm25 = None
            return

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
