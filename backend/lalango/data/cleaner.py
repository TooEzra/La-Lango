# lalango/data/cleaner.py
#
# Text cleaning utilities.
#
# Raw text data from the internet or community sources is often messy.
# This file contains functions to normalize and clean text before
# it goes into the tokenizer or model.
#
# "Garbage in, garbage out" — clean data leads to better translations.

import re
import unicodedata


def normalize_unicode(text):
    """
    Normalize unicode characters to a standard form.

    Many languages have multiple ways to encode the same character.
    For example, an accented character can be stored as:
      - One composed character (NFC): é  (single codepoint)
      - Two separate characters (NFD): e + combining accent (two codepoints)
    This function converts everything to NFC so the tokenizer sees them as the same.

    Args:
        text (str): Input text.

    Returns:
        str: Unicode-normalized text.
    """
    return unicodedata.normalize("NFC", text)


def remove_extra_whitespace(text):
    """
    Collapse multiple spaces into one and strip leading/trailing whitespace.

    Args:
        text (str): Input text, e.g. "hello   world  "

    Returns:
        str: Cleaned text, e.g. "hello world"
    """
    # Replace any sequence of whitespace characters (spaces, tabs, etc.)
    # with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_html_tags(text):
    """
    Remove any HTML tags from text.

    Some data sources (e.g. scraped websites) contain HTML like <b>hello</b>.
    We strip those out because the model should not learn HTML syntax.

    Args:
        text (str): Input text.

    Returns:
        str: Text with HTML tags removed.

    Example:
        >>> remove_html_tags("<b>Hello</b> world")
        "Hello world"
    """
    return re.sub(r"<[^>]+>", "", text)


def lowercase(text):
    """
    Convert text to lowercase.

    Note: Only use this for languages where case does not carry meaning.
    Some languages (e.g. German nouns) use case meaningfully,
    so think before applying this.

    Args:
        text (str): Input text.

    Returns:
        str: Lowercased text.
    """
    return text.lower()


def clean_sentence(text, do_lowercase=False, do_remove_html=True):
    """
    Apply all cleaning steps to a single sentence.

    This is the main function you will call in the preprocessing pipeline.
    It runs all the cleaning steps in the right order.

    Args:
        text (str): Raw input sentence.
        do_lowercase (bool): Whether to lowercase the text. Default: False.
                             Turn this on only if your language is case-insensitive.
        do_remove_html (bool): Whether to strip HTML tags. Default: True.

    Returns:
        str: The cleaned sentence.

    Example:
        >>> clean_sentence("  Hello   <b>World</b>!  ")
        "Hello World!"
    """
    # Step 1: Normalize unicode so accented characters are consistent
    text = normalize_unicode(text)

    # Step 2: Remove HTML tags if present
    if do_remove_html:
        text = remove_html_tags(text)

    # Step 3: Collapse extra whitespace
    text = remove_extra_whitespace(text)

    # Step 4: Optionally lowercase
    if do_lowercase:
        text = lowercase(text)

    return text


def clean_parallel_corpus(source_sentences, target_sentences, min_length=1, max_length=200):
    """
    Clean a full parallel corpus (lists of source and target sentences).

    In addition to cleaning each sentence, this function also filters out
    sentence pairs that are too short or too long — those tend to be noise.

    Args:
        source_sentences (list of str): Source language sentences.
        target_sentences (list of str): Target language sentences.
        min_length (int): Minimum number of characters. Shorter sentences are removed.
        max_length (int): Maximum number of characters. Longer sentences are removed.

    Returns:
        tuple: (cleaned_source, cleaned_target) — two lists of the same length.

    Example:
        >>> src = ["Hello!", "  Hi   ", ""]
        >>> tgt = ["Namaste!", "Kem cho", ""]
        >>> clean_parallel_corpus(src, tgt)
        (["Hello!", "Hi"], ["Namaste!", "Kem cho"])
    """
    # Make sure both lists have the same number of sentences
    assert len(source_sentences) == len(target_sentences), (
        f"Source has {len(source_sentences)} sentences but "
        f"target has {len(target_sentences)}. They must match."
    )

    cleaned_source = []
    cleaned_target = []
    skipped = 0

    for src, tgt in zip(source_sentences, target_sentences):
        # Clean each sentence
        src = clean_sentence(src)
        tgt = clean_sentence(tgt)

        # Skip pairs where either sentence is empty or out of length bounds
        if len(src) < min_length or len(tgt) < min_length:
            skipped += 1
            continue
        if len(src) > max_length or len(tgt) > max_length:
            skipped += 1
            continue

        cleaned_source.append(src)
        cleaned_target.append(tgt)

    print(f"Cleaned corpus: kept {len(cleaned_source)} pairs, skipped {skipped}")
    return cleaned_source, cleaned_target
