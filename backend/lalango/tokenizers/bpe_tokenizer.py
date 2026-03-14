# lalango/tokenizers/bpe_tokenizer.py
#
# BPE (Byte Pair Encoding) Tokenizer  —  Phase 3
#
# What is BPE and why do we need it?
#
#   The character tokenizer works, but it creates very long sequences.
#   For example, the word "translation" becomes 11 separate tokens.
#   Long sequences are slow to train and harder for the model to learn from.
#
#   BPE finds a smarter vocabulary by looking at the training data and
#   repeatedly merging the most common pairs of characters/tokens into one.
#
#   Example of how BPE builds its vocabulary step by step:
#
#     Start:   ["h","e","l","l","o"]  ["h","e","l","p"]
#     Step 1:  "he" appears most often → merge into one token
#              ["he","l","l","o"]  ["he","l","p"]
#     Step 2:  "hel" appears most often → merge
#              ["hel","l","o"]  ["hel","p"]
#     ...and so on until we reach our vocabulary size limit.
#
#   The result is a vocabulary of subword pieces that are longer than
#   characters but shorter than full words.
#
# This is a Phase 3 contribution. If you want to implement this:
#   1. Read the original BPE paper: https://arxiv.org/abs/1508.07909
#   2. Check the experiments/03_bpe_tokenizer.ipynb notebook for a walkthrough
#   3. Look at the char_tokenizer.py for the interface you need to match
#   4. Open a GitHub issue with the label `data` before you start


def get_word_frequencies(sentences):
    """
    Count how often each word appears in the training data.

    BPE starts by treating each word as a sequence of characters
    (with a special end-of-word marker </w>). We need to know
    how often each word appears so we can prioritize common patterns.

    Args:
        sentences (list of str): Training sentences.

    Returns:
        dict: Maps each word (as a tuple of characters) to its frequency.
              e.g. {('h','e','l','l','o','</w>'): 42}

    Example:
        >>> freqs = get_word_frequencies(["hello world", "hello there"])
        >>> freqs[('h','e','l','l','o','</w>')]
        2

    TODO (Phase 3):
        - Split each sentence into words (by spaces)
        - For each word, split it into individual characters
        - Add a '</w>' marker at the end of each word (this helps the model
          learn where words end)
        - Count how many times each character-sequence appears
    """
    raise NotImplementedError(
        "get_word_frequencies is not implemented yet. "
        "This is a Phase 3 task — see the docstring for guidance."
    )


def get_pair_frequencies(word_frequencies):
    """
    Count how often each adjacent pair of tokens appears across all words.

    This is the core of BPE — we find the most common pair and merge it.

    Args:
        word_frequencies (dict): Output from get_word_frequencies().

    Returns:
        dict: Maps each pair of tokens to its frequency.
              e.g. {('h','e'): 120, ('e','l'): 95}

    TODO (Phase 3):
        - For each word in word_frequencies:
            - Look at every adjacent pair of tokens in that word
            - Add (count of that word) to the pair's frequency
        - Return the full pair frequency dictionary
    """
    raise NotImplementedError(
        "get_pair_frequencies is not implemented yet. "
        "This is a Phase 3 task — see the docstring for guidance."
    )


def merge_pair(pair, word_frequencies):
    """
    Merge all occurrences of a token pair into a single token.

    Once we find the most common pair, we merge it everywhere.

    Args:
        pair (tuple): The pair to merge, e.g. ('h', 'e')
        word_frequencies (dict): Current word frequencies.

    Returns:
        dict: Updated word frequencies with the pair merged.
              e.g. ('h','e','l','l','o') becomes ('he','l','l','o')

    TODO (Phase 3):
        - For each word, scan for the pair
        - Replace every occurrence of the pair with a single merged token
        - Return the updated frequency dict
    """
    raise NotImplementedError(
        "merge_pair is not implemented yet. "
        "This is a Phase 3 task — see the docstring for guidance."
    )


class BPETokenizer:
    """
    A Byte Pair Encoding tokenizer.

    This is the Phase 3 upgrade from CharTokenizer.
    The interface (build, encode, decode) is intentionally the same
    so it can be swapped in without changing training code.

    TODO (Phase 3): Implement the build(), encode(), and decode() methods
    using the helper functions above.
    """

    def __init__(self, vocab_size=1000):
        # vocab_size controls how many merge operations we do.
        # More merges = longer subword units = shorter sequences.
        # A good starting range for low-resource languages: 500 – 4000
        self.vocab_size = vocab_size
        self.merges = []          # List of merge rules, in order they were learned
        self.token_to_idx = {}    # Final vocabulary mapping
        self.idx_to_token = {}

    def build(self, sentences):
        """
        Learn BPE merge rules from training sentences and build the vocabulary.

        TODO (Phase 3):
            1. Call get_word_frequencies() to get starting frequencies
            2. Add all individual characters to the vocabulary first
            3. Repeat until vocab_size is reached:
               a. Call get_pair_frequencies() to find the best pair
               b. Record the merge rule in self.merges
               c. Call merge_pair() to apply it
               d. Add the new merged token to the vocabulary
        """
        raise NotImplementedError("BPETokenizer.build() is not yet implemented.")

    def encode(self, sentence, add_sos=False, add_eos=True):
        """
        Apply learned BPE merges to encode a sentence into token indices.

        TODO (Phase 3):
            1. Split the sentence into words
            2. For each word, start with individual characters + '</w>'
            3. Apply merge rules in order (self.merges) wherever they match
            4. Look up each final token in self.token_to_idx
            5. Wrap with SOS/EOS if requested
        """
        raise NotImplementedError("BPETokenizer.encode() is not yet implemented.")

    def decode(self, indices, remove_special_tokens=True):
        """
        Convert token indices back into a readable sentence.

        TODO (Phase 3):
            1. Look up each index in self.idx_to_token
            2. Remove the '</w>' end-of-word markers
            3. Join everything back into a string
        """
        raise NotImplementedError("BPETokenizer.decode() is not yet implemented.")
