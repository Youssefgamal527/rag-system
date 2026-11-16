
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

        if not collection_exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )

        # Create text index for full-text search
        collection_info = self.client.get_collection(
            collection_name=self.collection_name
        )

        payload_indexes = collection_info.payload_schema or {}

        if "text" not in payload_indexes:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="text",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True
                ),
                wait=True
            )

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]]
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        if not chunks:
            return

        points = []

        for chunk, embedding in zip(chunks, embeddings):

            if len(embedding) != self.vector_size:
                raise ValueError(
                    f"Expected vector size {self.vector_size}, "
                    f"got {len(embedding)}"
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

    def replace_source(
        self,
        source: str,
        chunks: list[dict],
        embeddings: list[list[float]]
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        for embedding in embeddings:
            if len(embedding) != self.vector_size:
                raise ValueError(
                    f"Expected vector size {self.vector_size}, "
                    f"got {len(embedding)}"
                )

        for chunk in chunks:
            if chunk["metadata"]["source"] != source:
                raise ValueError(
                    "All chunks must belong to the specified source"
                )

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source)
                        )
                    ]
                )
            ),
            wait=True
        )

        self.add_chunks(chunks, embeddings)

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5
    ) -> list[dict]:

        if len(query_embedding) != self.vector_size:
            raise ValueError(
                f"Expected vector size {self.vector_size}, "
                f"got {len(query_embedding)}"
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

    def search_by_text(
        self,
        text: str,
        limit: int = 10
    ) -> list[dict]:

        if not text.strip():
            raise ValueError("Search text cannot be empty")

        if limit <= 0:
            raise ValueError("Limit must be greater than zero")

        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="text",
                        match=models.MatchText(text=text)
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        return [
            {
                "chunk_id": point.payload["chunk_id"],
                "text": point.payload["text"],
                "metadata": {
                    "source": point.payload["source"],
                    "page": point.payload["page"],
                    "chunk_index": point.payload["chunk_index"]
                }
            }
            for point in results
        ]

    def get_all_chunks(self) -> list[dict]:

        all_chunks = []

        records, next_offset = self.client.scroll(
            collection_name=self.collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False
        )

        while True:

            for point in records:

                payload = point.payload

                all_chunks.append({
                    "chunk_id": payload["chunk_id"],
                    "text": payload["text"],
                    "metadata": {
                        "source": payload["source"],
                        "page": payload["page"],
                        "chunk_index": payload["chunk_index"]
                    }
                })

            if next_offset is None:
                break

            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=next_offset,
                with_payload=True,
                with_vectors=False
            )

        return all_chunks
