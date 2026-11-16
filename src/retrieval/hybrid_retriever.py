
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.reranker import RRFFusion


class HybridRetriever:

    COUNTRY_ALIASES = {
        "ألمانيا": ["ألمانيا", "المانيا", "germany"],
        "إسبانيا": ["إسبانيا", "اسبانيا", "spain"],
        "إيطاليا": ["إيطاليا", "ايطاليا", "italy"],
        "المجر": ["المجر", "هنغاريا", "hungary"],
        "بولندا": ["بولندا", "poland"],
        "تركيا": ["تركيا", "turkey"],
        "جورجيا": ["جورجيا", "georgia"]
    }

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf: RRFFusion,
        vector_store=None
    ):

        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf = rrf
        self.vector_store = vector_store

    def _detect_countries(self, query: str) -> list[str]:

        query_lower = query.lower()

        return [
            country
            for country, aliases in self.COUNTRY_ALIASES.items()
            if any(alias in query_lower for alias in aliases)
        ]

    def _get_country_rows(
        self,
        query: str
    ) -> list[dict]:

        if self.vector_store is None:
            return []

        countries = self._detect_countries(query)

        country_rows = []
        seen_ids = set()

        for country in countries:

            results = self.vector_store.search_by_text(
                text=country,
                limit=100
            )

            for result in results:

                text = result.get("text", "")

                if "الدولة:" not in text:
                    continue

                if country not in text:
                    continue

                chunk_id = result.get("chunk_id")

                if chunk_id in seen_ids:
                    continue

                seen_ids.add(chunk_id)
                country_rows.append(result)

        return country_rows

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

        # 1. BM25 Search
        bm25_results = self.bm25_retriever.search(
            query=query,
            limit=retrieval_limit
        )

        # 2. Dense Search
        dense_results = self.dense_retriever.search(
            query=query,
            limit=retrieval_limit
        )

        # 3. RRF Fusion
        final_results = self.rrf.fuse(
            result_lists=[
                bm25_results,
                dense_results
            ],
            limit=limit
        )

        # 4. Direct country-row retrieval
        country_rows = self._get_country_rows(query)

        # 5. Merge without duplicates
        merged_results = []
        seen_ids = set()

        for result in country_rows + final_results:

            chunk_id = result.get("chunk_id")

            if chunk_id in seen_ids:
                continue

            seen_ids.add(chunk_id)
            merged_results.append(result)

        return merged_results
