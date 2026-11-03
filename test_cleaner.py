from src.text_cleaner import clean_text


def test_clean_text_removes_extra_spaces():
    text = "Hello     World"
    result = clean_text(text)

    assert result == "Hello World"


def test_clean_text_removes_extra_newlines():
    text = "Hello\n\n\n\nWorld"
    result = clean_text(text)

    assert result == "Hello\n\nWorld"


def test_clean_text_removes_invisible_characters():
    text = "Hello\u200b World\ufeff"
    result = clean_text(text)

    assert result == "Hello World"


def test_clean_text_removes_spaces_before_punctuation():
    text = "Hello , World !"
    result = clean_text(text)

    assert result == "Hello, World!"


def test_clean_text_strips_whitespace():
    text = "   Hello World   "
    result = clean_text(text)

    assert result == "Hello World"
