def split_text(
    text: str,
    chunk_size: int = 1800,
    chunk_overlap: int = 300
) -> list[str]:

    if chunk_size <= 0 or chunk_overlap < 0:
        raise ValueError("Invalid chunk size or overlap")

    if chunk_overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    separators = [
        "\n\n",
        "\n",
        "؟ ",
        ". ",
        "! ",
        "؛ ",
        "، ",
        " "
    ]

    while start < len(text):

        end = min(start + chunk_size, len(text))

        if end < len(text):

            window = text[start:end]

            # Find the nearest suitable text boundary
            boundary = -1

            for separator in separators:
                position = window.rfind(separator)

                if position >= chunk_size * 0.5:
                    boundary = max(
                        boundary,
                        position + len(separator)
                    )

            if boundary > 0:
                end = start + boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Move backward to preserve overlapping context
        next_start = end - chunk_overlap

        while next_start < end and text[next_start].isspace():
            next_start += 1

        start = next_start

    return chunks



def create_chunks(pages: list[dict], source: str) -> list[dict]:

    all_chunks = []

    for page in pages:

        text_chunks = split_text(page["text"])

        for index, chunk in enumerate(text_chunks, start=1):

            all_chunks.append({
                "chunk_id": f"{source}_p{page['page']}_c{index}",
                "text": chunk,
                "metadata": {
                    "source": source,
                    "page": page["page"],
                    "chunk_index": index
                }
            })

    return all_chunks