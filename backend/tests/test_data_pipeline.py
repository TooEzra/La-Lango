# tests/test_data_pipeline.py
#
# Unit tests for data cleaning, splitting, and loading.

from lalango.data.cleaner import (
    normalize_unicode,
    remove_extra_whitespace,
    remove_html_tags,
    clean_sentence,
    clean_parallel_corpus,
)
from lalango.data.splitter import split_corpus


class TestCleaner:

    def test_remove_extra_whitespace(self):
        assert remove_extra_whitespace("hello   world") == "hello world"
        assert remove_extra_whitespace("  hi  ") == "hi"
        assert remove_extra_whitespace("a  b  c") == "a b c"

    def test_remove_html_tags(self):
        assert remove_html_tags("<b>hello</b>") == "hello"
        assert remove_html_tags("hello <br/> world") == "hello  world"
        assert remove_html_tags("no tags here") == "no tags here"

    def test_clean_sentence_strips_html_and_whitespace(self):
        raw = "  <b>Hello</b>   world  "
        cleaned = clean_sentence(raw)
        assert cleaned == "Hello world"

    def test_clean_sentence_lowercase_option(self):
        assert clean_sentence("Hello World", do_lowercase=True) == "hello world"
        assert clean_sentence("Hello World", do_lowercase=False) == "Hello World"

    def test_clean_parallel_corpus_removes_empty_sentences(self):
        src = ["Hello!", "", "How are you?"]
        tgt = ["Namaste!", "", "Koso asa?"]
        cleaned_src, cleaned_tgt = clean_parallel_corpus(src, tgt)
        # The empty pair should be removed
        assert len(cleaned_src) == 2
        assert len(cleaned_tgt) == 2

    def test_clean_parallel_corpus_removes_long_sentences(self):
        src = ["Short", "A" * 300]  # second sentence is too long
        tgt = ["Short target", "B" * 300]
        cleaned_src, cleaned_tgt = clean_parallel_corpus(src, tgt, max_length=200)
        assert len(cleaned_src) == 1
        assert cleaned_src[0] == "Short"

    def test_clean_parallel_corpus_stays_aligned(self):
        """After cleaning, source and target lists must stay the same length."""
        src = ["Hello", "", "How are you?", "A" * 300]
        tgt = ["Hola",  "", "¿Cómo estás?", "B" * 300]
        cleaned_src, cleaned_tgt = clean_parallel_corpus(src, tgt, max_length=200)
        assert len(cleaned_src) == len(cleaned_tgt)

    def test_normalize_unicode_composes_characters(self):
        # NFD form: 'e' + combining accent = é (two codepoints)
        nfd = "e\u0301"
        # NFC form: é (one codepoint)
        nfc = "\xe9"
        # After normalization they should be the same
        assert normalize_unicode(nfd) == nfc


class TestSplitter:

    def setup_method(self):
        """Create a small dataset to test with."""
        self.src = [f"source sentence {i}" for i in range(100)]
        self.tgt = [f"target sentence {i}" for i in range(100)]

    def test_correct_number_of_splits(self):
        splits = split_corpus(self.src, self.tgt)
        # 80 / 10 / 10 split on 100 sentences
        assert len(splits["train"]["source"]) == 80
        assert len(splits["val"]["source"]) == 10
        assert len(splits["test"]["source"]) == 10

    def test_source_and_target_stay_aligned(self):
        """Source sentence i should still correspond to target sentence i after split."""
        splits = split_corpus(self.src, self.tgt, shuffle=False)
        for split in ["train", "val", "test"]:
            for src, tgt in zip(splits[split]["source"], splits[split]["target"]):
                # The sentence number should match between source and target
                src_num = src.split()[-1]
                tgt_num = tgt.split()[-1]
                assert src_num == tgt_num, (
                    f"Source '{src}' is paired with target '{tgt}' — they should match."
                )

    def test_no_data_lost(self):
        """The total number of sentences should be the same before and after splitting."""
        splits = split_corpus(self.src, self.tgt)
        total = (
            len(splits["train"]["source"]) +
            len(splits["val"]["source"]) +
            len(splits["test"]["source"])
        )
        assert total == 100

    def test_shuffle_with_same_seed_is_reproducible(self):
        """Same seed should always produce the same split."""
        splits1 = split_corpus(self.src, self.tgt, shuffle=True, seed=42)
        splits2 = split_corpus(self.src, self.tgt, shuffle=True, seed=42)
        assert splits1["train"]["source"] == splits2["train"]["source"]

    def test_different_seeds_produce_different_splits(self):
        splits1 = split_corpus(self.src, self.tgt, shuffle=True, seed=1)
        splits2 = split_corpus(self.src, self.tgt, shuffle=True, seed=99)
        # It is astronomically unlikely these are the same
        assert splits1["train"]["source"] != splits2["train"]["source"]

    def test_custom_ratios(self):
        splits = split_corpus(self.src, self.tgt, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        assert len(splits["train"]["source"]) == 70
        assert len(splits["val"]["source"]) == 20
        assert len(splits["test"]["source"]) == 10
