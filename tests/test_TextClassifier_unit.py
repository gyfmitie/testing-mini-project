from src.TextClassifier import TextClassifier


def test_preprocess_text_basic():
    """Unit test for basic text preprocessing."""
    classifier = TextClassifier()
    assert classifier.preprocess_text("HELLO WORLD") == "hello world"
    # Test case for mixed case text
    assert classifier.preprocess_text("PYthon is FUN") == "python is fun"
    # Test case for punctuation removal
    assert classifier.preprocess_text("Hello, World!") == "hello world"
    # Test case for numbers and symbols
    assert (
        classifier.preprocess_text("Numbers 123 and symbols!@#")
        == "numbers and symbols"
    )
    # Test case for leading/trailing spaces and multiple spaces
    assert (
        classifier.preprocess_text("  leading and trailing spaces  ")
        == "leading and trailing spaces"
    )
    # Test case for multiple spaces between words
    assert (
        classifier.preprocess_text("text with    multiple   spaces")
        == "text with multiple spaces"
    )


def test_preprocess_text_empty():
    """Unit test for empty string preprocessing."""
    classifier = TextClassifier()
    assert classifier.preprocess_text("") == ""
