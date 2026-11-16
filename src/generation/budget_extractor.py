
import re


class BudgetExtractor:
    """
    Extracts structured budget figures from retrieved document chunks.
    """

    COUNTRY_ALIASES = {
        "ألمانيا": ["ألمانيا", "المانيا", "germany"],
        "إسبانيا": ["إسبانيا", "اسبانيا", "spain"],
        "إيطاليا": ["إيطاليا", "ايطاليا", "italy"],
        "المجر": ["المجر", "هنغاريا", "hungary"],
        "بولندا": ["بولندا", "poland"],
        "تركيا": ["تركيا", "turkey"],
        "جورجيا": ["جورجيا", "georgia"],
    }

    NUMBER = r"\d[\d,\s]*"

    def _normalize_number(self, value: str) -> float:
        normalized = re.sub(r"[\s,]", "", value)
        return float(normalized)

    def _find_range(self, text: str, pattern: str):
        match = re.search(pattern, text, re.IGNORECASE)

        if not match:
            return None

        return (
            self._normalize_number(match.group(1)),
            self._normalize_number(match.group(2))
        )

    def _get_source(self, chunk: dict) -> dict:
        metadata = chunk.get("metadata", {})

        return {
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "chunk_id": chunk.get("chunk_id")
        }

    def extract(self, chunks: list[dict], country: str) -> dict:
        if country not in self.COUNTRY_ALIASES:
            raise ValueError(f"Unsupported country: {country}")

        country_chunks = []

        for chunk in chunks:
            text = chunk.get("text", "")
            text_lower = text.lower()

            if any(
                alias.lower() in text_lower
                for alias in self.COUNTRY_ALIASES[country]
            ):
                country_chunks.append(chunk)

        result = {
            "country": country,
            "currency": "EUR" if country == "ألمانيا" else None,
            "monthly_living": None,
            "public_semester_contribution": None,
            "annual_public_tuition": None,
            "annual_private_tuition": None,
            "sources": {}
        }

        for chunk in country_chunks:
            text = chunk.get("text", "")
            source = self._get_source(chunk)

            if result["monthly_living"] is None:
                living = self._find_range(
                    text,
                    rf"معيشة\s*شهريا.{{0,30}}?({self.NUMBER})\s*[–-]\s*({self.NUMBER})"
                )

                if living:
                    result["monthly_living"] = {
                        "min": living[0],
                        "max": living[1]
                    }
                    result["sources"]["monthly_living"] = source

            if result["public_semester_contribution"] is None:
                contribution = self._find_range(
                    text,
                    rf"({self.NUMBER})\s*[–-]\s*({self.NUMBER})\s*يورو\s*[/]?\s*فصل"
                )

                if contribution:
                    result["public_semester_contribution"] = {
                        "min": contribution[0],
                        "max": contribution[1],
                        "frequency": "per_semester"
                    }
                    result["sources"]["public_semester_contribution"] = source

            if result["annual_private_tuition"] is None:
                private = self._find_range(
                    text,
                    rf"خاص\s*سنويا.{{0,30}}?({self.NUMBER})\s*[–-]\s*({self.NUMBER})"
                )

                if private:
                    result["annual_private_tuition"] = {
                        "min": private[0],
                        "max": private[1]
                    }
                    result["sources"]["annual_private_tuition"] = source

        if result["public_semester_contribution"]:
            contribution = result["public_semester_contribution"]

            result["annual_public_tuition"] = {
                "min": contribution["min"] * 2,
                "max": contribution["max"] * 2,
                "note": "Calculated from two semesters"
            }

            result["sources"]["annual_public_tuition"] = (
                result["sources"]["public_semester_contribution"]
            )

        return result
