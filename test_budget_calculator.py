import pytest

from src.generation.budget_calculator import BudgetCalculator


@pytest.fixture
def calculator():
    return BudgetCalculator()


def test_public_university_budget(calculator):
    result = calculator.calculate(
        monthly_living=(850, 1100),
        annual_tuition=(0, 0),
        tuition_type="public",
        annual_mandatory_fees=(200, 900)
    )

    assert result["annual_living"] == {
        "min": 10200,
        "max": 13200
    }
    assert result["partial_first_year_total"] == {
        "min": 10400,
        "max": 14100
    }
    assert result["one_time_expenses"]["included"] is False


def test_private_university_budget(calculator):
    result = calculator.calculate(
        monthly_living=(850, 1100),
        annual_tuition=(9000, 14000),
        tuition_type="private"
    )

    assert result["partial_first_year_total"] == {
        "min": 19200,
        "max": 27200
    }


def test_budget_with_one_time_expenses(calculator):
    result = calculator.calculate(
        monthly_living=(850, 1100),
        annual_tuition=(0, 0),
        tuition_type="public",
        annual_mandatory_fees=(200, 900),
        one_time_expenses=(500, 1000)
    )

    assert result["partial_first_year_total"] == {
        "min": 10900,
        "max": 15100
    }
    assert result["one_time_expenses"]["included"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "monthly_living": (-1, 100),
            "annual_tuition": (0, 0),
            "tuition_type": "public"
        },
        {
            "monthly_living": (100, 50),
            "annual_tuition": (0, 0),
            "tuition_type": "public"
        },
        {
            "monthly_living": (100, 200),
            "annual_tuition": (500, 100),
            "tuition_type": "private"
        }
    ]
)
def test_invalid_budget_values_raise_error(calculator, kwargs):
    with pytest.raises(ValueError):
        calculator.calculate(**kwargs)
