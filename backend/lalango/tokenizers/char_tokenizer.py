# lalango/tokenizers/char_tokenizer.py
#
# A character-level tokenizer.
#
# What is a tokenizer?
#   Before a model can read text, we need to convert characters/words into numbers.
#   A tokenizer does exactly that — it builds a vocabulary (a mapping from
#   characters to numbers) and then uses it to encode and decode text.
#
# Why character-level?
#   For low-resource languages, we often do not have enough data to build a
#   good word-level vocabulary. Characters are simpler — most languages have
#   fewer than a few hundred unique characters, so even a tiny dataset covers
#   the full vocabulary.
#
# Example:
#   text = "hi"
#   encoded = [5, 12]   ← each character becomes a number
#   decoded = "hi"      ← we can convert back too

# Special tokens that every vocabulary needs:
#   <PAD> — used to pad short sentences to the same length as longer ones
#   <UNK> — used when we see a character we have never seen during training
#   <SOS> — Start Of Sentence, placed at the beginning of every target sentence
#   <EOS> — End Of Sentence, tells the decoder to stop generating
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


def build_vocabulary(sentences):
    """
    Build a character vocabulary from a list of sentences.

    This goes through every sentence, every character, and collects
    all the unique characters it finds. Then it assigns each one a number.

    Args:
        sentences (list of str): The training sentences to build vocab from.

    Returns:
        char_to_idx (dict): Maps each character to its number. e.g. {'a': 4, 'b': 5}
        idx_to_char (dict): The reverse — maps numbers back to characters. e.g. {4: 'a'}

    Example:
        >>> sentences = ["hello", "world"]
        >>> char_to_idx, idx_to_char = build_vocabulary(sentences)
        >>> char_to_idx['h']
        4
    """
    # Start with the four special tokens at fixed positions 0-3
    # so that PAD is always 0, UNK is always 1, etc.
    char_to_idx = {
        PAD_TOKEN: PAD_IDX,
        UNK_TOKEN: UNK_IDX,
        SOS_TOKEN: SOS_IDX,
        EOS_TOKEN: EOS_IDX,
    }

    # Go through every sentence, every character
    for sentence in sentences:
        for char in sentence:
            # Only add the character if we have not seen it before
            if char not in char_to_idx:
                # The next available index is just the current size of the dict
                char_to_idx[char] = len(char_to_idx)

    # Build the reverse mapping so we can go from numbers back to characters
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    return char_to_idx, idx_to_char


def encode(sentence, char_to_idx, add_sos=False, add_eos=True):
    """
    Convert a sentence (string) into a list of numbers.

    Args:
        sentence (str): The input text, e.g. "hello"
        char_to_idx (dict): The vocabulary mapping from build_vocabulary()
        add_sos (bool): Whether to prepend a Start Of Sentence token.
                        Set to True for target sentences during training.
        add_eos (bool): Whether to append an End Of Sentence token.
                        The decoder uses this to know when to stop.

    Returns:
        list of int: The encoded sentence, e.g. [4, 5, 6, 6, 7, 3]

    Example:
        >>> encoded = encode("hi", char_to_idx)
        >>> encoded
        [5, 12, 3]   ← 3 is EOS
    """
    # Convert each character to its index.
    # If the character is not in our vocabulary (e.g. a rare symbol),
    # use UNK_IDX as a fallback.
    indices = [char_to_idx.get(char, UNK_IDX) for char in sentence]

    # Optionally wrap with SOS and EOS tokens
    if add_sos:
        indices = [SOS_IDX] + indices
    if add_eos:
        indices = indices + [EOS_IDX]

    return indices


def decode(indices, idx_to_char, remove_special_tokens=True):
    """
    Convert a list of numbers back into a readable string.

    Args:
        indices (list of int): The encoded sentence, e.g. [4, 5, 6, 3]
        idx_to_char (dict): The reverse vocabulary from build_vocabulary()
        remove_special_tokens (bool): If True, strip out PAD, SOS, EOS tokens.
                                      Set to False if you want to see them.

    Returns:
        str: The decoded text, e.g. "hel"

    Example:
        >>> decoded = decode([5, 12, 3], idx_to_char)
        >>> decoded
        "hi"
    """
    special_token_indices = {PAD_IDX, SOS_IDX, EOS_IDX}

    chars = []
    for idx in indices:
        # Stop decoding when we hit EOS — the sentence is complete
        if idx == EOS_IDX:
            break

        # Skip special tokens if requested
        if remove_special_tokens and idx in special_token_indices:
            continue

        # Look up the character. If the index is unknown, use a placeholder.
        char = idx_to_char.get(idx, UNK_TOKEN)
        chars.append(char)

    return "".join(chars)


def pad_sequence(sequences, pad_idx=PAD_IDX):
    """
    Pad a batch of sequences so they all have the same length.

    Neural networks process data in batches, and all sequences in a batch
    must be the same length. We pad shorter sequences with PAD tokens.

    Args:
        sequences (list of list of int): A batch of encoded sentences.
        pad_idx (int): The index to use for padding (default: PAD_IDX = 0).

    Returns:
        list of list of int: All sequences padded to the length of the longest one.

    Example:
        >>> seqs = [[1, 2, 3], [1, 2]]
        >>> pad_sequence(seqs)
        [[1, 2, 3], [1, 2, 0]]
    """
    # Find the length of the longest sequence in the batch
    max_length = max(len(seq) for seq in sequences)

    # Pad every sequence that is shorter than max_length
    padded = []
    for seq in sequences:
        # How many PAD tokens do we need to add?
        padding_needed = max_length - len(seq)
        padded_seq = seq + [pad_idx] * padding_needed
        padded.append(padded_seq)

    return padded


class CharTokenizer:
    """
    A convenient class that bundles the vocabulary and encode/decode functions together.

    Instead of passing char_to_idx and idx_to_char everywhere,
    you create one CharTokenizer object and use it throughout your code.

    Example usage:
        tokenizer = CharTokenizer()
        tokenizer.build(["hello world", "how are you"])

        encoded = tokenizer.encode("hello")
        decoded = tokenizer.decode(encoded)
    """

    def __init__(self):
        # These will be populated when you call .build()
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def build(self, sentences):
        """Build the vocabulary from a list of training sentences."""
        self.char_to_idx, self.idx_to_char = build_vocabulary(sentences)
        self.vocab_size = len(self.char_to_idx)
        print(f"Vocabulary built: {self.vocab_size} unique characters")

    def encode(self, sentence, add_sos=False, add_eos=True):
        """Encode a sentence into a list of indices."""
        return encode(sentence, self.char_to_idx, add_sos=add_sos, add_eos=add_eos)

    def decode(self, indices, remove_special_tokens=True):
        """Decode a list of indices back into a string."""
        return decode(indices, self.idx_to_char, remove_special_tokens=remove_special_tokens)

    def vocab_size(self):
        """Return the number of unique characters in the vocabulary."""
        return len(self.char_to_idx)
