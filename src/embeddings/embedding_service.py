from src.embeddings.base import BaseEmbeddingProvider


class EmbeddingService:

    def __init__(self, provider: BaseEmbeddingProvider):
        self.provider = provider

    def generate_embedding(self, text: str) -> list[float]:

        if not text.strip():
            raise ValueError("Text cannot be empty")

        return self.provider.embed(text)

    def generate_embeddings(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("Texts cannot contain empty strings")

        return self.provider.embed_batch(texts)