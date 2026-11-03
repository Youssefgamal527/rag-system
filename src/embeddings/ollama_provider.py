import requests

from src.embeddings.base import BaseEmbeddingProvider


class OllamaEmbeddingProvider(BaseEmbeddingProvider):

    def __init__(
        self,
        model: str = "bge-m3",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, text: str) -> list[float]:

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": text
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data["embeddings"][0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:

        if not texts:
            return []

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": texts
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data["embeddings"]