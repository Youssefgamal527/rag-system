class RRFFusion:

    def __init__(self, k: int = 60):

        if k <= 0:
            raise ValueError("k must be greater than zero")

        self.k = k

    def fuse(
        self,
        result_lists: list[list[dict]],
        limit: int = 5
    ) -> list[dict]:

        if limit <= 0:
            raise ValueError("Limit must be greater than zero")

        fused_scores = {}
        documents = {}

        for results in result_lists:

            for rank, result in enumerate(results, start=1):

                chunk_id = result["chunk_id"]

                if chunk_id not in fused_scores:

                    fused_scores[chunk_id] = 0.0
                    documents[chunk_id] = result

                fused_scores[chunk_id] += 1 / (self.k + rank)

        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True
        )

        fused_results = []

        for chunk_id in ranked_ids[:limit]:

            result = documents[chunk_id].copy()

            result["rrf_score"] = fused_scores[chunk_id]

            fused_results.append(result)

        return fused_results