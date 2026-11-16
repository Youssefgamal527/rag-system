
import re


ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")
LATIN_PATTERN = re.compile(r"[a-zA-Z]")
NUMBER_PATTERN = re.compile(r"^[0-9٠-٩]+([.,/-][0-9٠-٩]+)*$")

OPENING_PUNCTUATION = {"(", "[", "{", "«", '"'}
CLOSING_PUNCTUATION = {
    ")", "]", "}", "»", '"', ".", ",", "،",
    "؛", ":", "!", "?", "؟", "%"
}


def _contains_arabic(text: str) -> bool:
    return bool(ARABIC_PATTERN.search(text))


def _contains_latin(text: str) -> bool:
    return bool(LATIN_PATTERN.search(text))


def _is_number(text: str) -> bool:
    return bool(NUMBER_PATTERN.fullmatch(text))


def _reorder_line(words: list[tuple]) -> str:

    if not words:
        return ""

    ordered = sorted(words, key=lambda word: word[0], reverse=True)

    tokens = [
        word[4].strip()
        for word in ordered
        if word[4].strip()
    ]

    index = 0

    while index < len(tokens):

        if not (
            _contains_latin(tokens[index])
            or _is_number(tokens[index])
        ):
            index += 1
            continue

        start = index

        while index < len(tokens) and (
            _contains_latin(tokens[index])
            or _is_number(tokens[index])
        ):
            index += 1

        tokens[start:index] = reversed(tokens[start:index])

    result = []

    for token in tokens:

        if token in CLOSING_PUNCTUATION:
            if result:
                result[-1] += token
            else:
                result.append(token)

        elif token in OPENING_PUNCTUATION:
            result.append(token)

        elif result and result[-1] in OPENING_PUNCTUATION:
            result[-1] += token

        else:
            result.append(token)

    return " ".join(result)


def _reconstruct_words(words: list[tuple]) -> str:

    if not words:
        return ""

    lines = {}

    for word in words:

        block_number = word[5]
        line_number = word[6]

        key = (block_number, line_number)

        lines.setdefault(key, []).append(word)

    reconstructed_blocks = []
    current_block = []
    previous_block_number = None

    for (block_number, line_number), line_words in sorted(lines.items()):

        if (
            previous_block_number is not None
            and block_number != previous_block_number
        ):
            reconstructed_blocks.append("\n".join(current_block))
            current_block = []

        line_text = _reorder_line(line_words)

        if line_text:
            current_block.append(line_text)

        previous_block_number = block_number

    if current_block:
        reconstructed_blocks.append("\n".join(current_block))

    return "\n\n".join(reconstructed_blocks)


def _cell_text(page, cell) -> str:

    if cell is None:
        return ""

    rect = page.parent[0].rect if False else None

    import pymupdf

    text = page.get_text(
        "text",
        clip=pymupdf.Rect(cell)
    )

    return re.sub(r"\s+", " ", text).strip()


def _cell_width(cell) -> float:

    if cell is None:
        return 0

    return cell[2] - cell[0]


def _cell_overlap(cell_a, cell_b) -> float:

    if cell_a is None or cell_b is None:
        return 0

    left = max(cell_a[0], cell_b[0])
    right = min(cell_a[2], cell_b[2])

    return max(0, right - left)


def _extract_table_text(page, table) -> str:
    """
    Extract a table while preserving column relationships.

    Header labels are matched to data cells using their
    horizontal coordinates, allowing merged cells.
    """

    rows = table.rows

    if not rows:
        return ""

    # 1. Find the header row.
    header_index = None

    for row_index, row in enumerate(rows):

        valid_cells = [
            cell
            for cell in row.cells
            if cell is not None and _cell_width(cell) >= 15
            and _cell_text(page, cell)
        ]

        if len(valid_cells) >= 5:
            header_index = row_index
            break

    if header_index is None:
        return ""

    header_cells = []

    # 2. Extract header labels.
    for cell in rows[header_index].cells:

        if cell is None or _cell_width(cell) < 15:
            continue

        text = _cell_text(page, cell)

        if text:
            header_cells.append({
                "rect": cell,
                "text": text
            })

    if not header_cells:
        return ""

    # 3. Merge a possible second header row.
    if header_index + 1 < len(rows):

        next_row = rows[header_index + 1]

        next_values = []

        for cell in next_row.cells:

            if cell is None or _cell_width(cell) < 15:
                continue

            text = _cell_text(page, cell)

            if text:
                next_values.append((cell, text))

        if len(next_values) <= 2:

            for cell, text in next_values:

                best_header = None
                best_overlap = 0

                for header in header_cells:

                    overlap = _cell_overlap(
                        cell,
                        header["rect"]
                    )

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_header = header

                if best_header is not None and best_overlap > 0:

                    best_header["text"] += " " + text

            data_start = header_index + 2

        else:
            data_start = header_index + 1

    else:
        data_start = header_index + 1

    # 4. Extract data rows and match cells to headers.
    output_rows = []

    for row in rows[data_start:]:

        row_values = []

        for cell in row.cells:

            if cell is None:
                continue

            text = _cell_text(page, cell)

            if not text:
                continue

            best_header = None
            best_overlap = 0

            for header in header_cells:

                overlap = _cell_overlap(
                    cell,
                    header["rect"]
                )

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_header = header

            if best_header is None or best_overlap <= 0:
                continue

            row_values.append(
                f"{best_header['text']}: {text}"
            )

        if row_values:
            output_rows.append(" | ".join(row_values))

    return "\n".join(output_rows)


def reconstruct_page_text(page) -> str:
    """
    Extract normal text and structured tables from a PDF page.
    """

    words = page.get_text("words")

    if not words:
        return ""

    # 1. Detect tables.
    try:
        tables = page.find_tables().tables
    except Exception:
        tables = []

    table_rectangles = []
    structured_tables = []

    for table in tables:

        table_rectangles.append(table.bbox)

        table_text = _extract_table_text(page, table)

        if table_text:
            structured_tables.append(table_text)

    # 2. Remove words belonging to detected tables.
    normal_words = []

    for word in words:

        center_x = (word[0] + word[2]) / 2
        center_y = (word[1] + word[3]) / 2

        inside_table = any(
            rect[0] <= center_x <= rect[2]
            and rect[1] <= center_y <= rect[3]
            for rect in table_rectangles
        )

        if not inside_table:
            normal_words.append(word)

    # 3. Reconstruct normal page text.
    normal_text = _reconstruct_words(normal_words)

    # 4. Append structured tables.
    if structured_tables:

        normal_text += "\n\n"

        normal_text += "\n\n".join(
            structured_tables
        )

    return normal_text
