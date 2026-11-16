
import re
import requests


class OllamaLLM:

    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
        num_predict: int = 512
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_predict = num_predict

    def _detect_language(self, text: str) -> str:

        arabic_chars = len(
            re.findall(r"[\u0600-\u06FF]", text)
        )

        latin_chars = len(
            re.findall(r"[a-zA-Z]", text)
        )

        if arabic_chars > latin_chars:
            return "Arabic"

        return "English"

    def _request(self, prompt: str) -> str:

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": self.num_predict,
                    "temperature": 0.1
                }
            },
            timeout=(10, self.timeout)
        )

        response.raise_for_status()

        result = response.json()
        answer = result.get("response", "").strip()

        if not answer:
            raise RuntimeError(
                f"Ollama returned an empty response: {result}"
            )

        return answer

    def generate(self, prompt: str) -> str:

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Generate the initial answer.
        answer = self._request(prompt)

        # Detect the requested language from the question.
        question_match = re.search(
            r"USER QUESTION:\s*(.*?)\s*FINAL ANSWER:",
            prompt,
            re.DOTALL
        )

        if not question_match:
            return answer

        question = question_match.group(1).strip()
        expected_language = self._detect_language(question)

        # Check the generated answer language.
        actual_language = self._detect_language(answer)

        if actual_language == expected_language:
            return answer

        # Ask the model to rewrite the answer in the correct language.
        correction_prompt = f"""
Rewrite the answer below entirely in {expected_language}.

STRICT RULES:
- Use {expected_language} only.
- Preserve every factual statement.
- Preserve all numbers, ranges, currencies and units.
- Do not add new information.
- Do not remove important information.
- Return only the corrected answer.

Original question:
{question}

Original answer:
{answer}

Corrected answer:
"""

        corrected_answer = self._request(correction_prompt)

        return corrected_answer
