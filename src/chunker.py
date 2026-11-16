
import re



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

    # Keep complete words and their original whitespace.
    tokens = re.findall(r"\S+", text)

    chunks = []
    start = 0

    while start < len(tokens):

        current_words = []
        current_length = 0
        end = start

        while end < len(tokens):

            word = tokens[end]
            extra_length = len(word)

            if current_words:
                extra_length += 1

            if current_length + extra_length > chunk_size:
                break

            current_words.append(word)
            current_length += extra_length
            end += 1

        if not current_words:
            raise ValueError("A single word exceeds chunk_size")

        chunks.append(" ".join(current_words))

        if end >= len(tokens):
            break

        # Move forward while retaining the requested overlap.
        next_start = max(start + 1, end - chunk_overlap)

        start = next_start

    return chunks


    separators = [
        r"\n\n",
        r"\n",
        r"(?<=[.!?؟])\s+",
        r"(?<=[؛;])\s+",
        r"(?<=[،,])\s+",
        r"\s+"
    ]

    def split_preserving_separator(
        content: str,
        separator: str
    ) -> list[str]:

        parts = re.split(f"({separator})", content)

        pieces = []

        for index in range(0, len(parts) - 1, 2):
            piece = parts[index] + parts[index + 1]

            if piece.strip():
                pieces.append(piece)

        if len(parts) % 2 and parts[-1].strip():
            pieces.append(parts[-1])

        return pieces

    def split_recursively(
        content: str,
        available_separators: list[str]
    ) -> list[str]:

        if len(content) <= chunk_size:
            return [content.strip()] if content.strip() else []

        if not available_separators:
            pieces = []
            current = ""

            for word in content.split():

                candidate = f"{current} {word}".strip()

                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        pieces.append(current)

                    if len(word) > chunk_size:
                        raise ValueError(
                            "A single word exceeds chunk_size"
                        )

                    current = word

            if current:
                pieces.append(current)

            return pieces

        separator = available_separators[0]
        remaining = available_separators[1:]

        parts = split_preserving_separator(content, separator)

        if len(parts) <= 1:
            return split_recursively(content, remaining)

        chunks = []
        current = ""

        for part in parts:

            candidate = current + part

            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())
                current = ""

            if len(part) > chunk_size:
                chunks.extend(
                    split_recursively(part, remaining)
                )
            else:
                current = part

        if current:
            chunks.append(current.strip())

        return chunks

    raw_chunks = split_recursively(text, separators)

    # Rebuild chunks with word-based overlap.
    final_chunks = []

    for chunk in raw_chunks:

        chunk = chunk.strip()

        if not chunk:
            continue

        if not final_chunks or chunk_overlap == 0:
            final_chunks.append(chunk)
            continue

        previous_words = final_chunks[-1].split()
        current_words = chunk.split()

        overlap_words = previous_words[-chunk_overlap:]

        # Keep the requested overlap whenever it fits.
        while overlap_words:

            combined = " ".join(
                overlap_words + current_words
            )

            if len(combined) <= chunk_size:
                break

            overlap_words.pop(0)

        if overlap_words:
            final_chunks.append(combined)
        else:
            final_chunks.append(chunk)

    return final_chunks


def create_chunks(
    pages: list[dict],
    source: str
) -> list[dict]:

    all_chunks = []

    for page in pages:

        text = page["text"].strip()

        # Detect structured country rows in tables.
        lines = text.splitlines()

        country_rows = []
        normal_lines = []

        for line in lines:

            if "الدولة:" in line:
                country_rows.append(line.strip())

            elif line.strip():
                normal_lines.append(line.strip())

        # Split normal text.
        text_chunks = split_text(
            "\n".join(normal_lines)
        )

        # Keep each country row in its own chunk.
        text_chunks.extend(country_rows)

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
