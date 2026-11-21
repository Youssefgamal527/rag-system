
import pytest

from src.generation.answer_generator import AnswerGenerator


class FakeRetriever:
    def search(self, query, limit, retrieval_limit):
        return [
            {
                "chunk_id": "test_001",
                "text": "RAG combines retrieval and generation.",
                "metadata": {
                    "source": "test.pdf",
                    "page": 1
                }
            }
        ]


class FakePromptBuilder:
    def build(self, query, retrieved_chunks):
        return f"Question: {query}\nContext: {retrieved_chunks}"


class FakeLLM:
    def generate(self, prompt):
        return "Test answer."


@pytest.fixture
def generator():
    return AnswerGenerator(
        retriever=FakeRetriever(),
        prompt_builder=FakePromptBuilder(),
        llm=FakeLLM()
    )


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "germany_001",
            "text": (
                "معيشة شهريا: 850 – 1,100 يورو "
                "| الدولة: ألمانيا"
            ),
            "metadata": {"source": "guide.pdf", "page": 38}
        },
        {
            "chunk_id": "spain_001",
            "text": (
                "معيشة شهريا: 900 – 1,360 يورو "
                "| الدولة: إسبانيا"
            ),
            "metadata": {"source": "guide.pdf", "page": 38}
        },
        {
            "chunk_id": "italy_001",
            "text": (
                "معيشة شهريا: 700 – 1,700 يورو "
                "| الدولة: إيطاليا"
            ),
            "metadata": {"source": "guide.pdf", "page": 38}
        },
        {
            "chunk_id": "poland_001",
            "text": (
                "معيشة شهريا: 330 – 900 يورو "
                "| الدولة: بولندا"
            ),
            "metadata": {"source": "guide.pdf", "page": 38}
        }
    ]


def test_generator_returns_answer(generator):
    result = generator.generate("What is RAG?")

    assert result["answer"] == "Test answer."

    assert result["sources"] == [
        {"source": "test.pdf", "page": 1}
    ]

    assert len(result["retrieved_chunks"]) == 1


def test_detect_countries_supports_arabic_and_english(generator):
    assert generator._detect_countries(
        "تكلفة الدراسة في ألمانيا"
    ) == ["ألمانيا"]

    assert generator._detect_countries(
        "Study costs in Germany"
    ) == ["ألمانيا"]


def test_monthly_living_query_detection(generator):
    assert generator._is_monthly_living_query(
        "ما هي تكلفة المعيشة الشهرية في ألمانيا؟"
    )

    assert not generator._is_monthly_living_query(
        "ما هي رسوم الجامعات الخاصة في ألمانيا؟"
    )


def test_casual_monthly_query_detection(generator):
    assert generator._is_monthly_living_query(
        "محتاج كام كل شهر عشان أعيش في ألمانيا؟"
    )


def test_monthly_comparison_query_detection(generator):
    query = "إيه أرخص دولة من ناحية المعيشة الشهرية؟"

    assert generator._is_monthly_living_query(query)


def test_filter_context_preserves_chunks_for_unrelated_query(
    generator,
    sample_chunks
):
    result = generator._filter_context(
        "ما هي شروط القبول والتأشيرة؟",
        sample_chunks
    )

    assert result == sample_chunks


def test_filter_context_selects_germany_private_tuition_only(
    generator,
    sample_chunks
):
    result = generator._filter_context(
        "ما هي تكلفة الدراسة السنوية في الجامعات الخاصة في ألمانيا؟",
        sample_chunks
    )

    assert len(result) == 1
    assert result[0]["chunk_id"] == "germany_001"
    assert "الدولة: ألمانيا" in result[0]["text"]


def test_filter_context_selects_only_compared_countries(
    generator,
    sample_chunks
):
    query = (
        "قارن المعيشة الشهرية في ألمانيا وإسبانيا وإيطاليا"
    )

    result = generator._filter_context(query, sample_chunks)

    result_ids = {
        chunk["chunk_id"] for chunk in result
    }

    assert result_ids == {
        "germany_001",
        "spain_001",
        "italy_001"
    }

    assert "poland_001" not in result_ids


def test_total_budget_query_is_not_treated_as_tuition_only(generator):
    query = "ما إجمالي ميزانية الدراسة والمعيشة في ألمانيا؟"

    assert not generator._is_monthly_living_query(query)
    assert generator._detect_countries(query) == ["ألمانيا"]
