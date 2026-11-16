
class BudgetCalculator:
    """
    Calculates first-year study and living budget
    from explicitly provided, structured figures.
    """

    def calculate(
        self,
        monthly_living: tuple[float, float],
        annual_tuition: tuple[float, float],
        tuition_type: str,
        one_time_expenses: tuple[float, float] | None = None
    ) -> dict:

        if len(monthly_living) != 2:
            raise ValueError("Monthly living must contain min and max")

        if len(annual_tuition) != 2:
            raise ValueError("Annual tuition must contain min and max")

        if one_time_expenses is not None and len(one_time_expenses) != 2:
            raise ValueError("One-time expenses must contain min and max")

        living_min, living_max = monthly_living
        tuition_min, tuition_max = annual_tuition

        if one_time_expenses is None:
            expenses_min = 0
            expenses_max = 0
            includes_one_time_expenses = False
        else:
            expenses_min, expenses_max = one_time_expenses
            includes_one_time_expenses = True

        values = [
            living_min, living_max,
            tuition_min, tuition_max,
            expenses_min, expenses_max
        ]

        if any(value < 0 for value in values):
            raise ValueError("Budget values cannot be negative")

        if living_min > living_max:
            raise ValueError("Living minimum cannot exceed maximum")

        if tuition_min > tuition_max:
            raise ValueError("Tuition minimum cannot exceed maximum")

        if expenses_min > expenses_max:
            raise ValueError("Expenses minimum cannot exceed maximum")

        annual_living_min = living_min * 12
        annual_living_max = living_max * 12

        total_min = annual_living_min + tuition_min + expenses_min
        total_max = annual_living_max + tuition_max + expenses_max

        return {
            "tuition_type": tuition_type,
            "currency": "EUR",
            "annual_living": {
                "min": annual_living_min,
                "max": annual_living_max
            },
            "annual_tuition": {
                "min": tuition_min,
                "max": tuition_max
            },
            "one_time_expenses": {
                "min": expenses_min,
                "max": expenses_max,
                "included": includes_one_time_expenses
            },
            "partial_first_year_total": {
                "min": total_min,
                "max": total_max
            }
        }
