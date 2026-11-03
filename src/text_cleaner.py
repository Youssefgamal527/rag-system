import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving its meaning.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove invisible characters
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # Replace non-breaking spaces
    text = text.replace("\u00a0", " ")

    # Normalize spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([،؛,.!?;:])", r"\1", text)

    # Add space between Arabic letters and numbers
    text = re.sub(r"(?<=[\u0600-\u06FF])(?=\d)", " ", text)

    # Remove excessive empty lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
