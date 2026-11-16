import json


class AnswerGenerator:

    COUNTRY_ALIASES = {
        "ألمانيا": ["ألمانيا", "المانيا", "germany"],
        "إسبانيا": ["إسبانيا", "اسبانيا", "spain"],
        "إيطاليا": ["إيطاليا", "ايطاليا", "italy"],
        "المجر": ["المجر", "هنغاريا", "hungary"],
        "بولندا": ["بولندا", "poland"],
        "تركيا": ["تركيا", "turkey"],
        "جورجيا": ["جورجيا", "georgia"]
    }

    MONTHLY_TERMS = [
        "معيشة شهريا",
        "المعيشة شهريا",
        "المعيشة الشهرية",
        "مصاريف المعيشة",
        "تكاليف المعيشة",
        "monthly living",
        "living expenses",
        "living costs"
    ]

    TOTAL_BUDGET_TERMS = [
        "السنة الأولى كاملة",
        "إجمالي الميزانية",
        "إجمالي التكلفة",
        "إجمالي ميزانية",
        "ميزانية الدراسة والمعيشة",
        "التكلفة الكاملة",
        "التكلفة الإجمالية",
        "total budget",
        "first year"
    ]

    PRIVATE_TUITION_TERMS = [
        "الجامعات الخاصة",
        "جامعة خاصة",
        "رسوم الجامعات الخاصة",
        "مصاريف الجامعات الخاصة",
        "private universities",
        "private university",
        "private tuition"
    ]

    PUBLIC_TUITION_TERMS = [
        "الجامعات الحكومية",
        "جامعة حكومية",
        "رسوم الجامعات الحكومية",
        "مصاريف الجامعات الحكومية",
        "public universities",
        "public university",
        "public tuition"
    ]

    def __init__(
        self,
        retriever,
        prompt_builder,
        llm,
        budget_extractor=None,
        budget_calculator=None
    ):
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.budget_extractor = budget_extractor
        self.budget_calculator = budget_calculator

    def _detect_countries(self, query: str) -> list[str]:
        query_lower = query.lower()

        return [
            country
            for country, aliases in self.COUNTRY_ALIASES.items()
            if any(alias in query_lower for alias in aliases)
        ]

    def _contains_term(self, query: str, terms: list[str]) -> bool:
        query_lower = query.lower()
        return any(term in query_lower for term in terms)

    def _is_monthly_living_query(self, query: str) -> bool:
        return self._contains_term(query, self.MONTHLY_TERMS)

    def _is_total_budget_query(self, query: str) -> bool:
        return self._contains_term(query, self.TOTAL_BUDGET_TERMS)

    def _is_private_tuition_query(self, query: str) -> bool:
        return self._contains_term(query, self.PRIVATE_TUITION_TERMS)

    def _is_public_tuition_query(self, query: str) -> bool:
        return self._contains_term(query, self.PUBLIC_TUITION_TERMS)

    def _unique_chunks(self, chunks: list[dict]) -> list[dict]:
        unique = []
        seen_ids = set()

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")

            if chunk_id is None:
                unique.append(chunk)
            elif chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique.append(chunk)

        return unique

    def _filter_country_rows(
        self,
        countries: list[str],
        chunks: list[dict]
    ) -> list[dict]:

        selected = []

        for country in countries:
            aliases = self.COUNTRY_ALIASES[country]

            for chunk in chunks:
                text = chunk.get("text", "")
                text_lower = text.lower()

                country_found = any(
                    alias in text_lower
                    for alias in aliases
                )

                if country_found and "الدولة:" in text:
                    selected.append(chunk)

        return self._unique_chunks(selected)

    def _filter_context(
        self,
        query: str,
        chunks: list[dict]
    ) -> list[dict]:

        countries = self._detect_countries(query)

        if self._is_total_budget_query(query):
            if "ألمانيا" in countries:
                relevant_pages = {7, 24, 38}

                selected = [
                    chunk
                    for chunk in chunks
                    if chunk.get("metadata", {}).get("page")
                    in relevant_pages
                ]

                return self._unique_chunks(selected) if selected else chunks

            return chunks

        if not countries:
            return chunks

        if (
            self._is_private_tuition_query(query)
            or self._is_public_tuition_query(query)
        ):
            selected = self._filter_country_rows(countries, chunks)
            return selected if selected else chunks

        if self._is_monthly_living_query(query):
            selected = self._filter_country_rows(countries, chunks)
            return selected if selected else chunks

        return chunks

    def _build_budget_estimates(
        self,
        query: str,
        chunks: list[dict]
    ) -> dict | None:

        countries = self._detect_countries(query)

        if not self._is_total_budget_query(query):
            return None

        if "ألمانيا" not in countries:
            return None

        if not self.budget_extractor or not self.budget_calculator:
            return None

        extracted = self.budget_extractor.extract(
            chunks=chunks,
            country="ألمانيا"
        )

        living = extracted.get("monthly_living")
        public_fees = extracted.get(
            "annual_public_semester_contribution"
        )
        private_tuition = extracted.get("annual_private_tuition")

        if not living or not public_fees or not private_tuition:
            return None

        monthly_living = (
            living["min"],
            living["max"]
        )

        public_result = self.budget_calculator.calculate(
            monthly_living=monthly_living,
            annual_tuition=(0, 0),
            tuition_type="public",
            annual_mandatory_fees=(
                public_fees["min"],
                public_fees["max"]
            )
        )

        private_result = self.budget_calculator.calculate(
            monthly_living=monthly_living,
            annual_tuition=(
                private_tuition["min"],
                private_tuition["max"]
            ),
            tuition_type="private"
        )

        return {
            "country": "ألمانيا",
            "currency": extracted["currency"],
            "public_university": public_result,
            "private_university": private_result,
            "extraction_sources": extracted["sources"]
        }

    def _format_budget_answer(self, budget: dict) -> str:

        currency = budget.get("currency", "EUR")

        def format_range(values):
            minimum = values["min"]
            maximum = values["max"]
            return f"{minimum:,.0f} - {maximum:,.0f} {currency}"

        public = budget["public_university"]
        private = budget["private_university"]

        return f"""
إجمالي ميزانية السنة الأولى للدراسة والمعيشة في {budget['country']}:

**أولًا: الجامعات الحكومية**

- مصروفات المعيشة السنوية:
  {format_range(public['annual_living'])}

- الرسوم الدراسية السنوية:
  {format_range(public['annual_tuition'])}

- الرسوم الإلزامية السنوية:
  {format_range(public['annual_mandatory_fees'])}

- إجمالي ميزانية السنة الأولى:
  {format_range(public['partial_first_year_total'])}

**ثانيًا: الجامعات الخاصة**

- مصروفات المعيشة السنوية:
  {format_range(private['annual_living'])}

- الرسوم الدراسية السنوية:
  {format_range(private['annual_tuition'])}

- الرسوم الإلزامية السنوية:
  {format_range(private['annual_mandatory_fees'])}

- إجمالي ميزانية السنة الأولى:
  {format_range(private['partial_first_year_total'])}

ملاحظة: هذه تقديرات أساسية، ولا تشمل المصروفات الإضافية غير المحتسبة أو تذاكر السفر. وقد تختلف الرسوم في بعض الجامعات الحكومية حسب الولاية والبرنامج الدراسي.
""".strip()

    def _build_budget_prompt(
        self,
        prompt: str,
        budget_estimates: dict
    ) -> str:

        budget_context = json.dumps(
            budget_estimates,
            ensure_ascii=False,
            indent=2
        )

        return prompt + f"""

==================================================
CALCULATED BUDGET DATA
==================================================

{budget_context}

Use the calculated data only for financial questions.
Do not modify or invent financial values.
Answer in the same language as the user's question.
"""

    def generate(
        self,
        query: str,
        limit: int = 10,
        retrieval_limit: int = 20
    ) -> dict:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        retrieved_chunks = self.retriever.search(
            query=query,
            limit=limit,
            retrieval_limit=retrieval_limit
        )

        if not retrieved_chunks:
            return {
                "answer": "No relevant information was found in the documents.",
                "sources": [],
                "retrieved_chunks": [],
                "budget_estimates": None
            }

        selected_chunks = self._filter_context(
            query=query,
            chunks=retrieved_chunks
        )

        budget_estimates = self._build_budget_estimates(
            query=query,
            chunks=selected_chunks
        )

        if budget_estimates:
            answer = self._format_budget_answer(budget_estimates)
        else:
            prompt = self.prompt_builder.build(
                query=query,
                retrieved_chunks=selected_chunks
            )
            answer = self.llm.generate(prompt)

        sources = []
        seen_sources = set()

        for chunk in selected_chunks:
            metadata = chunk.get("metadata", {})

            source = metadata.get("source")
            page = metadata.get("page")

            source_key = (source, page)

            if source_key not in seen_sources and (
                source is not None or page is not None
            ):
                seen_sources.add(source_key)

                sources.append({
                    "source": source,
                    "page": page
                })

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": selected_chunks,
            "budget_estimates": budget_estimates
        }
