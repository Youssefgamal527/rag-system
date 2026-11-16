
from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.ollama_provider import OllamaEmbeddingProvider

from src.vector_store.qdrant_store import QdrantStore

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import RRFFusion

from src.generation.ollama_llm import OllamaLLM
from src.generation.prompt_builder import PromptBuilder
from src.generation.answer_generator import AnswerGenerator
from src.generation.budget_extractor import BudgetExtractor
from src.generation.budget_calculator import BudgetCalculator

from src.ingestion_pipeline import IngestionPipeline


class RAGApplication:

    def __init__(self):

        # 1. Embedding components
        self.embedding_provider = OllamaEmbeddingProvider(
            model="bge-m3"
        )

        self.embedding_service = EmbeddingService(
            provider=self.embedding_provider
        )

        # 2. Vector database
        self.vector_store = QdrantStore(
            collection_name="rag_documents",
            vector_size=1024
        )

        # 3. Retrieval components
        self.bm25_retriever = BM25Retriever()

        self.dense_retriever = DenseRetriever(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store
        )

        self.rrf = RRFFusion()

        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=self.bm25_retriever,
            dense_retriever=self.dense_retriever,
            rrf=self.rrf,
            vector_store=self.vector_store
        )

        # 4. Generation components
        self.llm = OllamaLLM(
            model="qwen2.5:3b-instruct"
        )

        self.prompt_builder = PromptBuilder()

        # Budget components
        self.budget_extractor = BudgetExtractor()
        self.budget_calculator = BudgetCalculator()

        self.answer_generator = AnswerGenerator(
            retriever=self.hybrid_retriever,
            prompt_builder=self.prompt_builder,
            llm=self.llm,
            budget_extractor=self.budget_extractor,
            budget_calculator=self.budget_calculator
        )

        # 5. Ingestion component
        self.ingestion_pipeline = IngestionPipeline(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store
        )

        self.initialized = False
        self.documents_count = 0

    def initialize(self) -> dict:

        # Ensure that the Qdrant collection exists
        self.vector_store.create_collection()

        # Load existing chunks from Qdrant
        chunks = self.vector_store.get_all_chunks()

        # Rebuild the in-memory BM25 index
        self.bm25_retriever.index_documents(chunks)

        self.documents_count = len(chunks)
        self.initialized = True

        return {
            "status": "Success",
            "chunks_loaded": self.documents_count
        }

    def ingest_pdf(self, file_path: str) -> dict:

        if not self.initialized:
            raise RuntimeError(
                "Application must be initialized first"
            )

        # Ingest the PDF into Qdrant
        result = self.ingestion_pipeline.ingest_pdf(file_path)

        # Synchronize BM25 with Qdrant
        if result["status"] == "Success":
            chunks = self.vector_store.get_all_chunks()

            self.bm25_retriever.index_documents(chunks)

            self.documents_count = len(chunks)

        return result

    def ask(self, query: str) -> dict:

        if not self.initialized:
            raise RuntimeError(
                "Application must be initialized first"
            )

        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Handle an empty knowledge base
        if self.documents_count == 0:
            return {
                "answer": "No documents have been added to the knowledge base yet.",
                "sources": [],
                "retrieved_chunks": []
            }

        return self.answer_generator.generate(
            query=query
        )
