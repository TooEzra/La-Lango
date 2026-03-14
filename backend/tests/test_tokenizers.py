# tests/test_tokenizers.py
#
# Unit tests for the tokenizers.
#
# Run all tests with:   pytest tests/
# Run just these tests: pytest tests/test_tokenizers.py -v

from lalango.tokenizers.char_tokenizer import (
    build_vocabulary,
    encode,
    decode,
    pad_sequence,
    CharTokenizer,
    PAD_IDX,
    UNK_IDX,
    SOS_IDX,
    EOS_IDX,
)


class TestBuildVocabulary:

    def test_special_tokens_are_always_present(self):
        """The four special tokens should always be in the vocabulary."""
        char_to_idx, _ = build_vocabulary(["hello"])
        assert "<PAD>" in char_to_idx
        assert "<UNK>" in char_to_idx
        assert "<SOS>" in char_to_idx
        assert "<EOS>" in char_to_idx

    def test_special_tokens_have_correct_indices(self):
        """Special tokens must always be at indices 0-3."""
        char_to_idx, _ = build_vocabulary(["hello"])
        assert char_to_idx["<PAD>"] == 0
        assert char_to_idx["<UNK>"] == 1
        assert char_to_idx["<SOS>"] == 2
        assert char_to_idx["<EOS>"] == 3

    def test_all_characters_in_vocabulary(self):
        """Every character in the training sentences should get a vocab entry."""
        sentences = ["abc", "def"]
        char_to_idx, _ = build_vocabulary(sentences)
        for char in "abcdef":
            assert char in char_to_idx, f"Character '{char}' missing from vocabulary"

    def test_no_duplicate_indices(self):
        """Each character should have a unique index."""
        char_to_idx, _ = build_vocabulary(["hello world", "foo bar"])
        indices = list(char_to_idx.values())
        assert len(indices) == len(set(indices)), "Duplicate indices found in vocabulary"

    def test_reverse_mapping_is_consistent(self):
        """idx_to_char should be the exact reverse of char_to_idx."""
        char_to_idx, idx_to_char = build_vocabulary(["test"])
        for char, idx in char_to_idx.items():
            assert idx_to_char[idx] == char

    def test_empty_sentences(self):
        """An empty list of sentences should still produce a vocabulary with special tokens."""
        char_to_idx, _ = build_vocabulary([])
        assert len(char_to_idx) == 4  # Just the 4 special tokens

    def test_duplicate_characters_not_duplicated(self):
        """Repeated characters across sentences should only appear once in vocab."""
        char_to_idx, _ = build_vocabulary(["aaa", "aaa", "bbb"])
        # 4 special tokens + 'a' + 'b' = 6
        assert len(char_to_idx) == 6


class TestEncode:

    def setup_method(self):
        """Set up a small vocabulary before each test."""
        self.char_to_idx, self.idx_to_char = build_vocabulary(["hello"])

    def test_encode_returns_list_of_ints(self):
        encoded = encode("hello", self.char_to_idx)
        assert isinstance(encoded, list)
        assert all(isinstance(i, int) for i in encoded)

    def test_eos_appended_by_default(self):
        """By default, EOS should be the last token."""
        encoded = encode("hi", self.char_to_idx, add_eos=True)
        assert encoded[-1] == EOS_IDX

    def test_sos_prepended_when_requested(self):
        """When add_sos=True, SOS should be the first token."""
        encoded = encode("hi", self.char_to_idx, add_sos=True)
        assert encoded[0] == SOS_IDX

    def test_no_eos_when_disabled(self):
        encoded = encode("hi", self.char_to_idx, add_eos=False)
        assert EOS_IDX not in encoded

    def test_unknown_characters_map_to_unk(self):
        """Characters not in the vocabulary should map to UNK_IDX."""
        encoded = encode("xyz123", self.char_to_idx)
        # x, y, z, 1, 2, 3 are not in "hello" vocabulary
        # All should be UNK_IDX (plus EOS at the end)
        for idx in encoded[:-1]:  # exclude EOS
            assert idx == UNK_IDX

    def test_empty_string(self):
        """Encoding an empty string with add_eos=True should return just [EOS_IDX]."""
        encoded = encode("", self.char_to_idx, add_eos=True)
        assert encoded == [EOS_IDX]


class TestDecode:

    def setup_method(self):
        self.char_to_idx, self.idx_to_char = build_vocabulary(["hello"])

    def test_decode_recovers_original_text(self):
        """Encoding then decoding should return the original string."""
        original = "hello"
        encoded = encode(original, self.char_to_idx, add_eos=False)
        decoded = decode(encoded, self.idx_to_char)
        assert decoded == original

    def test_eos_stops_decoding(self):
        """Decoding should stop at EOS and not include it in the output."""
        # Manually build a sequence with EOS in the middle
        encoded = encode("he", self.char_to_idx, add_eos=False)
        encoded_with_eos = encoded + [EOS_IDX] + encode("llo", self.char_to_idx, add_eos=False)
        decoded = decode(encoded_with_eos, self.idx_to_char)
        assert decoded == "he"

    def test_special_tokens_removed_by_default(self):
        """PAD and SOS tokens should not appear in the decoded output."""
        encoded = [SOS_IDX] + encode("hi", self.char_to_idx, add_eos=False) + [PAD_IDX]
        decoded = decode(encoded, self.idx_to_char, remove_special_tokens=True)
        assert "<SOS>" not in decoded
        assert "<PAD>" not in decoded


class TestPadSequence:

    def test_sequences_padded_to_max_length(self):
        sequences = [[1, 2, 3], [1, 2], [1]]
        padded = pad_sequence(sequences)
        # All should be length 3 (the max)
        assert all(len(seq) == 3 for seq in padded)

    def test_shortest_sequence_padded_correctly(self):
        sequences = [[1, 2, 3], [4]]
        padded = pad_sequence(sequences)
        assert padded[1] == [4, PAD_IDX, PAD_IDX]

    def test_equal_length_sequences_unchanged(self):
        sequences = [[1, 2], [3, 4], [5, 6]]
        padded = pad_sequence(sequences)
        assert padded == sequences

    def test_single_sequence_unchanged(self):
        sequences = [[1, 2, 3]]
        padded = pad_sequence(sequences)
        assert padded == [[1, 2, 3]]


class TestCharTokenizer:

    def test_full_pipeline(self):
        """A complete encode → decode roundtrip should work end to end."""
        tokenizer = CharTokenizer()
        tokenizer.build(["hello world", "how are you"])

        original = "hello"
        encoded = tokenizer.encode(original, add_eos=False)
        decoded = tokenizer.decode(encoded, remove_special_tokens=True)
        assert decoded == original

    def test_vocab_size_is_set_after_build(self):
        tokenizer = CharTokenizer()
        tokenizer.build(["abc"])
        # 4 special tokens + 3 characters = 7
        assert tokenizer.vocab_size == 7
