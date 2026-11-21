
import re


class AnswerGenerator:

    COUNTRY_ALIASES = {
        "ألمانيا": ["ألمانيا", "المانيا", "germany"],
        "إسبانيا": ["إسبانيا", "اسبانيا", "spain"],
        "إيطاليا": ["إيطاليا", "ايطاليا", "italy"],
        "المجر": ["المجر", "هنغاريا", "hungary"],
        "بولندا": ["بولندا", "poland"],
        "تركيا": ["تركيا", "turkey"],
        "جورجيا": ["جورجيا", "georgia"],
    }

    MONTHLY_TERMS = [
        "المعيشة شهريا",
        "المعيشة شهرياً",
        "المعيشة الشهرية",
        "تكلفة المعيشة",
        "المصاريف الشهرية",
        "تكاليف المعيشة",
        "living cost",
        "monthly cost",
        "per month",
        "monthly living",
        "living expenses",
    ]

    CASUAL_MONTHLY_TERMS = [
        "محتاج كام كل شهر", "أحتاج كام كل شهر",
        "محتاج كام شهريا", "اصرف كام شهريا",
        "أصرف كام شهريا", "كام هصرف في الشهر",
        "تكفيني كام شهريا", "المصاريف في الشهر",
        "كل شهر عشان اعيش", "شهريا عشان اعيش",
        "كل شهر", "شهريا", "شهرياً", "في الشهر",
    ]

    TOTAL_BUDGET_TERMS = [
        "الميزانية الكلية", "التكلفة الكلية",
        "التكلفة الإجمالية", "اجمالي التكلفة",
        "إجمالي التكلفة", "total budget",
        "total cost", "first year cost",
    ]

    PRIVATE_TUITION_TERMS = [
        "الجامعات الخاصة", "جامعة خاصة",
        "الجامعة الخاصة", "رسوم الجامعات الخاصة",
        "مصاريف الجامعات الخاصة",
        "private university", "private universities",
        "private tuition",
    ]

    PUBLIC_TUITION_TERMS = [
        "الجامعات الحكومية", "جامعة حكومية",
        "الجامعة الحكومية", "رسوم الجامعات الحكومية",
        "مصاريف الجامعات الحكومية",
        "public university", "public universities",
        "public tuition",
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

    @staticmethod
    def _contains_term(query, terms):
        query = query.lower()
        return any(term.lower() in query for term in terms)

    def _detect_countries(self, query):
        found = []

        for country, aliases in self.COUNTRY_ALIASES.items():
            if any(
                alias.lower() in query.lower()
                for alias in aliases
            ):
                found.append(country)

        return found

    def _is_monthly_living_query(self, query):
        return (
            self._contains_term(query, self.MONTHLY_TERMS)
            or self._contains_term(
                query,
                self.CASUAL_MONTHLY_TERMS
            )
        )

    def _is_total_budget_query(self, query):
        return self._contains_term(
            query,
            self.TOTAL_BUDGET_TERMS
        )

    def _is_private_tuition_query(self, query):
        return self._contains_term(
            query,
            self.PRIVATE_TUITION_TERMS
        )

    def _is_public_tuition_query(self, query):
        return self._contains_term(
            query,
            self.PUBLIC_TUITION_TERMS
        )

    @staticmethod
    def _unique_chunks(chunks):
        results = []
        seen = set()

        for chunk in chunks:
            key = chunk.get("chunk_id") or chunk.get("text", "")

            if key in seen:
                continue

            seen.add(key)
            results.append(chunk)

        return results

    def _filter_country_rows(self, query, chunks):
        countries = self._detect_countries(query)

        if not countries:
            return chunks

        selected = []

        for chunk in chunks:
            text = chunk.get("text", "").lower()

            if any(
                alias.lower() in text
                for country in countries
                for alias in self.COUNTRY_ALIASES[country]
            ):
                selected.append(chunk)

        return selected or chunks

    def _filter_context(self, query, chunks):
        chunks = self._unique_chunks(chunks)

        is_monthly = self._is_monthly_living_query(query)
        is_private = self._is_private_tuition_query(query)
        is_public = self._is_public_tuition_query(query)
        is_budget = self._is_total_budget_query(query)

        # الأسئلة غير الشهرية:
        # نفلتر حسب الدولة فقط في أسئلة الرسوم والميزانية.
        if not is_monthly:
            if is_private or is_public or is_budget:
                return self._filter_country_rows(query, chunks)

            return chunks

        # اختيار بيانات المعيشة الشهرية.
        monthly_chunks = [
            chunk for chunk in chunks
            if any(
                term in chunk.get("text", "").lower()
                for term in [
                    "المعيشة",
                    "شهريا",
                    "شهرياً",
                    "شهري",
                    "monthly",
                    "living cost",
                    "living expenses",
                ]
            )
        ]

        if monthly_chunks:
            chunks = monthly_chunks

        return self._filter_country_rows(query, chunks)

    # ==========================================
    # Budget calculation
    # ==========================================

    def _build_budget_estimates(self, query, chunks):

        if not self._is_total_budget_query(query):
            return None

        countries = self._detect_countries(query)

        if "ألمانيا" not in countries:
            return None

        if not self.budget_extractor or not self.budget_calculator:
            return None

        extracted = self.budget_extractor.extract(
            chunks=chunks,
            country="ألمانيا"
        )

        if not extracted:
            return None

        living = extracted.get("monthly_living")
        public_fees = extracted.get(
            "annual_public_semester_contribution"
        )
        private_tuition = extracted.get(
            "annual_private_tuition"
        )

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

    # ==========================================
    # Tuition extraction
    # ==========================================

    def _build_tuition_estimate(self, query, chunks):

        is_private = self._is_private_tuition_query(query)
        is_public = self._is_public_tuition_query(query)

        if not is_private and not is_public:
            return None

        countries = self._detect_countries(query)
        results = []

        for chunk in chunks:
            text = chunk.get("text", "")
            lower = text.lower()

            if countries and not any(
                alias.lower() in lower
                for country in countries
                for alias in self.COUNTRY_ALIASES[country]
            ):
                continue

            private = any(
                term.lower() in lower
                for term in self.PRIVATE_TUITION_TERMS
            )

            public = any(
                term.lower() in lower
                for term in self.PUBLIC_TUITION_TERMS
            )

            if is_private and not private:
                continue

            if is_public and not public:
                continue

            metadata = chunk.get("metadata", {})

            results.append({
                "text": text,
                "source": metadata.get("source"),
                "page": metadata.get("page"),
            })

        return results or None

    def _format_tuition_answer(self, estimate):

        lines = ["معلومات الرسوم الدراسية:"]

        for item in estimate:
            lines.append(item["text"])

            if item.get("source"):
                lines.append(
                    f"المصدر: {item['source']} "
                    f"- صفحة {item.get('page')}"
                )

        return "\n".join(lines)

    # ==========================================
    # Budget formatting
    # ==========================================

    @staticmethod
    def _format_range(data):
        return (
            f"{data['min']:,.0f} - "
            f"{data['max']:,.0f}"
        )

    def _format_budget_answer(self, estimates):

        if not isinstance(estimates, dict):
            return str(estimates)

        currency = estimates.get("currency", "EUR")

        lines = [
            "الميزانية السنوية التقديرية للدراسة في ألمانيا:",
            "",
            "أولاً: الجامعة الحكومية",
        ]

        public = estimates["public_university"]

        lines.append(
            "المعيشة السنوية: "
            + self._format_range(public["annual_living"])
            + f" {currency}"
        )

        lines.append(
            "المساهمات والرسوم الإلزامية: "
            + self._format_range(
                public["annual_mandatory_fees"]
            )
            + f" {currency}"
        )

        lines.append(
            "إجمالي السنة الأولى: "
            + self._format_range(
                public["partial_first_year_total"]
            )
            + f" {currency}"
        )

        lines.extend([
            "",
            "ثانياً: الجامعة الخاصة",
        ])

        private = estimates["private_university"]

        lines.append(
            "المعيشة السنوية: "
            + self._format_range(private["annual_living"])
            + f" {currency}"
        )

        lines.append(
            "الرسوم الدراسية السنوية: "
            + self._format_range(private["annual_tuition"])
            + f" {currency}"
        )

        lines.append(
            "إجمالي السنة الأولى: "
            + self._format_range(
                private["partial_first_year_total"]
            )
            + f" {currency}"
        )

        lines.extend([
            "",
            "ملاحظة: الأرقام تقديرية وتعتمد على البيانات "
            "المستخرجة من المستندات."
        ])

        return "\n".join(lines)

    # ==========================================
    # Monthly comparison
    # ==========================================

    def _extract_monthly_country_data(self, chunks):

        results = []
        seen_countries = set()

        for chunk in chunks:
            text = chunk.get("text", "")

            if "الدولة:" not in text:
                continue

            country = None

            for name, aliases in self.COUNTRY_ALIASES.items():
                if any(
                    alias.lower() in text.lower()
                    for alias in aliases
                ):
                    country = name
                    break

            if not country or country in seen_countries:
                continue

            match = re.search(
                r"(\d[\d,]*(?:\.\d+)?)\s*[-–—]\s*"
                r"(\d[\d,]*(?:\.\d+)?)",
                text
            )

            if not match:
                continue

            minimum = float(
                match.group(1).replace(",", "")
            )

            maximum = float(
                match.group(2).replace(",", "")
            )

            if minimum > maximum:
                continue

            lower = text.lower()

            if "يورو" in lower or "eur" in lower:
                currency = "EUR"
            elif "دولار" in lower or "usd" in lower:
                currency = "USD"
            else:
                currency = None

            metadata = chunk.get("metadata", {})

            results.append({
                "country": country,
                "min": minimum,
                "max": maximum,
                "currency": currency,
                "source": metadata.get("source"),
                "page": metadata.get("page"),
            })

            seen_countries.add(country)

        return results

    def _format_monthly_comparison(self, data):

        if len(data) < 2:
            return ""

        currencies = {
            item["currency"] for item in data
        }

        lines = [
            "مقارنة تكاليف المعيشة الشهرية:"
        ]

        for item in data:

            if item["currency"]:
                amount = (
                    f"{item['min']:,.0f} - "
                    f"{item['max']:,.0f} "
                    f"{item['currency']}"
                )
            else:
                amount = "العملة غير محددة في المصدر"

            lines.append(
                f"- {item['country']}: {amount}"
            )

        if None in currencies:
            lines.append(
                "بعض البيانات لا تحدد العملة، "
                "لذلك لا يمكن إجراء مقارنة مالية كاملة."
            )

        elif len(currencies) > 1:
            lines.append(
                "البيانات تستخدم عملات مختلفة، "
                "ولا يمكن ترتيب الدول دون تحويل العملات."
            )

        else:
            lowest = min(
                data,
                key=lambda item: item["min"]
            )

            lines.append(
                f"أقل حد أدنى في البيانات هو "
                f"{lowest['country']} بقيمة "
                f"{lowest['min']:,.0f} "
                f"{lowest['currency']} شهرياً."
            )

            lines.append(
                "قد تتداخل نطاقات التكاليف، "
                "لذلك لا يمكن الجزم بأنها الأرخص في كل الحالات."
            )

        return "\n".join(lines)

    # ==========================================
    # Main generation
    # ==========================================

    def generate(
        self,
        query,
        limit=10,
        retrieval_limit=20
    ):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        retrieved_chunks = self.retriever.search(
            query=query,
            limit=limit,
            retrieval_limit=retrieval_limit
        )

        if not retrieved_chunks:
            return {
                "answer": (
                    "No relevant information was found "
                    "in the documents."
                ),
                "sources": [],
                "retrieved_chunks": [],
                "budget_estimates": None,
                "tuition_estimate": None,
            }

        selected_chunks = self._filter_context(
            query=query,
            chunks=retrieved_chunks
        )

        budget_estimates = self._build_budget_estimates(
            query=query,
            chunks=selected_chunks
        )

        tuition_estimate = None

        if not budget_estimates:
            tuition_estimate = self._build_tuition_estimate(
                query=query,
                chunks=selected_chunks
            )

        monthly_comparison = None

        if self._is_monthly_living_query(query):

            monthly_data = self._extract_monthly_country_data(
                selected_chunks
            )

            monthly_comparison = self._format_monthly_comparison(
                monthly_data
            )

        if budget_estimates:

            answer = self._format_budget_answer(
                budget_estimates
            )

        elif tuition_estimate:

            answer = self._format_tuition_answer(
                tuition_estimate
            )

        elif monthly_comparison:

            answer = monthly_comparison

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

            if (
                source_key not in seen_sources
                and (source is not None or page is not None)
            ):
                seen_sources.add(source_key)

                sources.append({
                    "source": source,
                    "page": page,
                })

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": selected_chunks,
            "budget_estimates": budget_estimates,
            "tuition_estimate": tuition_estimate,
        }
