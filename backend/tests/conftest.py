# tests/conftest.py
#
# Shared fixtures for pytest.
# These are automatically available to all test files without importing.

import pytest
from lalango.tokenizers.char_tokenizer import CharTokenizer


@pytest.fixture
def small_corpus():
    """A tiny parallel corpus for testing data pipeline functions."""
    source = [
        "How are you?",
        "I am fine.",
        "Good morning.",
        "Thank you.",
        "See you later.",
    ]
    target = [
        "Koso asa?",
        "Hanv boro asa.",
        "Suprabhaat.",
        "Dev borem korum.",
        "Fuddem mellya.",
    ]
    return source, target


@pytest.fixture
def src_tokenizer(small_corpus):
    """A CharTokenizer built from the small corpus source sentences."""
    source, _ = small_corpus
    tokenizer = CharTokenizer()
    tokenizer.build(source)
    return tokenizer


@pytest.fixture
def tgt_tokenizer(small_corpus):
    """A CharTokenizer built from the small corpus target sentences."""
    _, target = small_corpus
    tokenizer = CharTokenizer()
    tokenizer.build(target)
    return tokenizer
