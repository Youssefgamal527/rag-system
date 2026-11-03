from qdrant_client import QdrantClient, models
from uuid import uuid5, NAMESPACE_DNS


class QdrantStore:

    def __init__(
        self,
        collection_name: str = "rag_documents",
        url: str = "http://localhost:6333",
        vector_size: int = 1024
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size

        self.client = QdrantClient(url=url)

    def create_collection(self) -> None:

        collections = self.client.get_collections().collections

        collection_exists = any(
            collection.name == self.collection_name
            for collection in collections
        )

        if collection_exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE
            )
        )

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]]
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        if not chunks:
            return

        points = []

        for chunk, embedding in zip(chunks, embeddings):

            if len(embedding) != self.vector_size:
                raise ValueError(
                    f"Expected vector size {self.vector_size}, got {len(embedding)}"
                )

            point_id = str(
                uuid5(NAMESPACE_DNS, chunk["chunk_id"])
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        **chunk["metadata"]
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5
    ) -> list[dict]:

        if len(query_embedding) != self.vector_size:
            raise ValueError(
                f"Expected vector size {self.vector_size}, got {len(query_embedding)}"
            )

        if limit <= 0:
            raise ValueError("Limit must be greater than zero")

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            with_payload=True
        ).points

        return [
            {
                "chunk_id": point.payload["chunk_id"],
                "text": point.payload["text"],
                "metadata": {
                    "source": point.payload["source"],
                    "page": point.payload["page"],
                    "chunk_index": point.payload["chunk_index"]
                },
                "score": point.score
            }
            for point in results
        ]