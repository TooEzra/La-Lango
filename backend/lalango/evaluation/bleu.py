# lalango/evaluation/bleu.py
#
# BLEU Score — implemented from scratch.
#
# Read docs/evaluation_guide.md for an explanation of what BLEU measures
# and how to interpret the scores.
#
# This implementation follows the original BLEU paper:
# "BLEU: a Method for Automatic Evaluation of Machine Translation"
# Papineni et al., 2002  https://aclanthology.org/P02-1040.pdf

import math
from collections import Counter


def get_ngrams(tokens, n):
    """
    Extract all n-grams from a list of tokens.

    An n-gram is a contiguous sequence of n tokens.

    Args:
        tokens (list): A list of characters or words, e.g. ['h','e','l','l','o']
        n (int): The n-gram size. n=1 gives single characters, n=2 gives pairs, etc.

    Returns:
        Counter: A count of how many times each n-gram appears.

    Example:
        >>> get_ngrams(['a', 'b', 'c'], 2)
        Counter({('a','b'): 1, ('b','c'): 1})
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        # Extract a tuple of n consecutive tokens starting at position i
        ngram = tuple(tokens[i: i + n])
        ngrams.append(ngram)
    return Counter(ngrams)


def clipped_precision(hypothesis, reference, n):
    """
    Compute the clipped n-gram precision.

    Precision = (matching n-grams) / (total n-grams in hypothesis)

    "Clipped" means we cap the count of each n-gram in the hypothesis
    by how many times it appears in the reference. This prevents a model
    from cheating by repeating a common word many times.

    Args:
        hypothesis (list): The model's translation (list of characters or words).
        reference (list): The human reference translation.
        n (int): The n-gram size.

    Returns:
        float: The clipped precision for this n-gram size.
    """
    # Count n-grams in hypothesis and reference
    hyp_ngrams = get_ngrams(hypothesis, n)
    ref_ngrams = get_ngrams(reference, n)

    # Count total n-grams in the hypothesis
    total_hyp_ngrams = sum(hyp_ngrams.values())

    if total_hyp_ngrams == 0:
        return 0.0

    # For each n-gram in the hypothesis, count how many are also in the reference
    # "Clipped" means we can only count up to the reference count — not more
    matching = 0
    for ngram, count in hyp_ngrams.items():
        # The reference count for this n-gram (0 if it does not appear)
        ref_count = ref_ngrams.get(ngram, 0)
        # We can "match" at most min(hypothesis_count, reference_count) copies
        matching += min(count, ref_count)

    return matching / total_hyp_ngrams


def brevity_penalty(hypothesis_length, reference_length):
    """
    Penalize translations that are too short.

    Without this, a model could get high precision by outputting just
    a single very common word. The brevity penalty reduces the score
    when the hypothesis is shorter than the reference.

    Args:
        hypothesis_length (int): Length of the model's translation.
        reference_length (int): Length of the reference translation.

    Returns:
        float: A value between 0 and 1. 1.0 means no penalty.
    """
    if hypothesis_length >= reference_length:
        # No penalty if the hypothesis is at least as long as the reference
        return 1.0
    else:
        # Exponential penalty that grows as the hypothesis gets shorter
        return math.exp(1 - reference_length / hypothesis_length)


def bleu_score(hypothesis, reference, max_n=4):
    """
    Compute the BLEU score between a hypothesis and a reference translation.

    BLEU combines precision scores for n-grams of size 1 through max_n,
    then applies the brevity penalty.

    Args:
        hypothesis (str or list): The model's translation.
                                  Can be a string (will be split into characters)
                                  or a list of tokens.
        reference (str or list): The human reference translation.
        max_n (int): Maximum n-gram size to consider. Default: 4.
                     For character-level models, using max_n=4 is standard.

    Returns:
        float: BLEU score from 0 to 100.

    Example:
        >>> bleu_score("How are you?", "How are you?")
        100.0
        >>> bleu_score("How are you", "How are you doing")
        # will be less than 100 due to brevity penalty and missing "doing"
    """
    # Convert strings to character lists if needed
    if isinstance(hypothesis, str):
        hypothesis = list(hypothesis)
    if isinstance(reference, str):
        reference = list(reference)

    # Edge case: empty hypothesis
    if len(hypothesis) == 0:
        return 0.0

    # Compute clipped precision for each n-gram size from 1 to max_n
    precisions = []
    for n in range(1, max_n + 1):
        # Skip if the hypothesis or reference is shorter than n
        if len(hypothesis) < n or len(reference) < n:
            break
        p = clipped_precision(hypothesis, reference, n)
        if p == 0:
            # If any precision is zero, BLEU is 0 (log(0) is undefined)
            return 0.0
        precisions.append(math.log(p))

    if not precisions:
        return 0.0

    # Geometric mean of all precision scores (average of log values)
    avg_log_precision = sum(precisions) / len(precisions)

    # Apply brevity penalty
    bp = brevity_penalty(len(hypothesis), len(reference))

    # Final BLEU score (multiplied by 100 for readability)
    score = bp * math.exp(avg_log_precision) * 100
    return round(score, 2)


def corpus_bleu(hypotheses, references, max_n=4):
    """
    Compute BLEU score over an entire test set (corpus-level).

    Corpus-level BLEU is more reliable than averaging sentence-level scores.
    It accumulates n-gram counts across all sentences before computing precision.

    Args:
        hypotheses (list of str or list of list): All model translations.
        references (list of str or list of list): All reference translations.
        max_n (int): Maximum n-gram size.

    Returns:
        float: Corpus-level BLEU score from 0 to 100.
    """
    assert len(hypotheses) == len(references), (
        "Number of hypotheses and references must match."
    )

    # Accumulate counts across the entire corpus
    total_hyp_len = 0
    total_ref_len = 0
    # matching_counts[n] = total matching n-grams across all sentences
    matching_counts = [0] * (max_n + 1)
    # total_counts[n] = total n-grams in all hypotheses
    total_counts = [0] * (max_n + 1)

    for hyp, ref in zip(hypotheses, references):
        if isinstance(hyp, str):
            hyp = list(hyp)
        if isinstance(ref, str):
            ref = list(ref)

        total_hyp_len += len(hyp)
        total_ref_len += len(ref)

        for n in range(1, max_n + 1):
            if len(hyp) < n:
                continue
            hyp_ngrams = get_ngrams(hyp, n)
            ref_ngrams = get_ngrams(ref, n)
            total_counts[n] += sum(hyp_ngrams.values())
            for ngram, count in hyp_ngrams.items():
                matching_counts[n] += min(count, ref_ngrams.get(ngram, 0))

    # Compute precision for each n
    precisions = []
    for n in range(1, max_n + 1):
        if total_counts[n] == 0:
            break
        if matching_counts[n] == 0:
            return 0.0
        precisions.append(math.log(matching_counts[n] / total_counts[n]))

    if not precisions:
        return 0.0

    avg_log_precision = sum(precisions) / len(precisions)
    bp = brevity_penalty(total_hyp_len, total_ref_len)
    score = bp * math.exp(avg_log_precision) * 100
    return round(score, 2)
