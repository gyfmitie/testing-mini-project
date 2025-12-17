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

# =============================================================================
# TDD EXERCISE: The following tests will FAIL with the current implementation.
# Students must update src/TextClassifier.py to make them pass.
# =============================================================================


def test_preprocess_text_html_tags():
    """TDD Exercise: HTML tags should be removed from text."""
    classifier = TextClassifier()
    assert classifier.preprocess_text("<p>Hello</p>") == "hello"
    assert classifier.preprocess_text("<div>Some <b>bold</b> text</div>") == "some bold text"
    assert classifier.preprocess_text("No tags here") == "no tags here"


def test_preprocess_text_urls():
    """TDD Exercise: URLs should be removed from text."""
    classifier = TextClassifier()
    assert classifier.preprocess_text("Visit https://example.com today") == "visit today"
    assert classifier.preprocess_text("Check http://test.org for info") == "check for info"
    assert classifier.preprocess_text("No URLs here") == "no urls here"


def test_preprocess_text_accented_characters():
    """TDD Exercise: Accented characters should be normalized to ASCII."""
    classifier = TextClassifier()
    assert classifier.preprocess_text("café") == "cafe"
    assert classifier.preprocess_text("résumé") == "resume"
    assert classifier.preprocess_text("naïve") == "naive"
    assert classifier.preprocess_text("El Niño") == "el nino"
