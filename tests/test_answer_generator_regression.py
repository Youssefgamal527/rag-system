
import pytest

from src.generation.answer_generator import AnswerGenerator


class FakeRetriever:
    def search(self, query, limit, retrieval_limit):
        return []


class FakePromptBuilder:
    def build(self, query, retrieved_chunks):
        return ""


class FakeLLM:
    def generate(self, prompt):
        return ""


@pytest.fixture
def generator():
    return AnswerGenerator(
        retriever=FakeRetriever(),
        prompt_builder=FakePromptBuilder(),
        llm=FakeLLM()
    )


@pytest.fixture
def country_chunks():
    return [
        {
            "chunk_id": "germany_001",
            "text": "الدولة: ألمانيا | معيشة شهريا: 850 – 1,100 يورو",
            "metadata": {"source": "guide.pdf", "page": 38}
        },
        {
            "chunk_id": "spain_001",
            "text": "الدولة: إسبانيا | معيشة شهريا: 900 – 1,360 يورو",
            "metadata": {"source": "guide.pdf", "page": 38}
        },
        {
            "chunk_id": "italy_001",
            "text": "الدولة: إيطاليا | معيشة شهريا: 700 – 1,700 يورو",
            "metadata": {"source": "guide.pdf", "page": 38}
        }
    ]


def test_arabic_country_comparison(generator, country_chunks):
    result = generator._filter_context(
        "قارن مصاريف المعيشة شهريا في ألمانيا وإسبانيا",
        country_chunks
    )

    assert len(result) == 2
    assert "ألمانيا" in result[0]["text"]
    assert "إسبانيا" in result[1]["text"]


def test_english_country_comparison(generator, country_chunks):
    result = generator._filter_context(
        "Compare monthly living expenses in Germany and Spain",
        country_chunks
    )

    assert len(result) == 2
    assert any("ألمانيا" in chunk["text"] for chunk in result)
    assert any("إسبانيا" in chunk["text"] for chunk in result)


def test_single_country_filtering(generator, country_chunks):
    result = generator._filter_context(
        "مصاريف المعيشة شهريا في ألمانيا",
        country_chunks
    )

    assert len(result) == 1
    assert "ألمانيا" in result[0]["text"]


def test_non_monthly_question_keeps_all_chunks(
    generator,
    country_chunks
):
    result = generator._filter_context(
        "ما شروط الدراسة في ألمانيا؟",
        country_chunks
    )

    assert result == country_chunks


def test_no_matching_country_chunk_preserves_context(generator):
    chunks = [
        {
            "chunk_id": "general_001",
            "text": "معلومات عامة عن الدراسة في الخارج",
            "metadata": {"source": "guide.pdf", "page": 5}
        }
    ]

    result = generator._filter_context(
        "مصاريف المعيشة شهريا في ألمانيا",
        chunks
    )

    assert result == chunks

def test_generate_calls_retriever_with_correct_arguments():
    retriever = FakeRetriever()
    generator = AnswerGenerator(
        retriever=retriever,
        prompt_builder=FakePromptBuilder(),
        llm=FakeLLM()
    )

    result = generator.generate(
        "What is Hybrid Search?",
        limit=5,
        retrieval_limit=15
    )

    assert result["answer"] == "No relevant information was found in the documents."
    assert result["sources"] == []
    assert result["retrieved_chunks"] == []


def test_generate_returns_llm_answer():
    chunks = [
        {
            "chunk_id": "chunk_001",
            "text": "Hybrid Search combines BM25 and Dense Retrieval.",
            "metadata": {
                "source": "rag_guide.pdf",
                "page": 5
            }
        }
    ]

    class Retriever:
        def search(self, query, limit, retrieval_limit):
            return chunks

    class LLM:
        def generate(self, prompt):
            return "Hybrid Search combines keyword and semantic retrieval."

    generator = AnswerGenerator(
        retriever=Retriever(),
        prompt_builder=FakePromptBuilder(),
        llm=LLM()
    )

    result = generator.generate("What is Hybrid Search?")

    assert result["answer"] == (
        "Hybrid Search combines keyword and semantic retrieval."
    )


def test_generate_extracts_sources_correctly():
    chunks = [
        {
            "chunk_id": "chunk_001",
            "text": "Information from page 5.",
            "metadata": {
                "source": "rag_guide.pdf",
                "page": 5
            }
        },
        {
            "chunk_id": "chunk_002",
            "text": "More information from page 5.",
            "metadata": {
                "source": "rag_guide.pdf",
                "page": 5
            }
        },
        {
            "chunk_id": "chunk_003",
            "text": "Information from page 8.",
            "metadata": {
                "source": "rag_guide.pdf",
                "page": 8
            }
        }
    ]

    class Retriever:
        def search(self, query, limit, retrieval_limit):
            return chunks

    generator = AnswerGenerator(
        retriever=Retriever(),
        prompt_builder=FakePromptBuilder(),
        llm=FakeLLM()
    )

    result = generator.generate("Explain Hybrid Search")

    assert result["sources"] == [
        {"source": "rag_guide.pdf", "page": 5},
        {"source": "rag_guide.pdf", "page": 8}
    ]


def test_generate_rejects_empty_query(generator):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        generator.generate("")

