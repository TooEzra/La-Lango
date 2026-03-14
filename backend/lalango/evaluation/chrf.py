# lalango/evaluation/chrf.py
#
# chrF Score (Character n-gram F-score)
#
# chrF is similar to BLEU but works at the character level instead of the word level.
# This makes it more suitable for:
#   - Morphologically rich languages (where words have many forms)
#   - Languages that do not use spaces to separate words
#   - Agglutinative languages (like Tulu, Finnish, Turkish)
#
# Reference: "chrF: character n-gram F-score for automatic MT evaluation"
#             Maja Popovic, 2015  https://aclanthology.org/W15-3049.pdf

from collections import Counter


def get_char_ngrams(text, n):
    """
    Extract character n-grams from a string.

    Args:
        text (str): Input text, e.g. "hello"
        n (int): n-gram size.

    Returns:
        Counter: Character n-gram counts, e.g. Counter({('h','e'): 1, ('e','l'): 1, ...})

    Example:
        >>> get_char_ngrams("abc", 2)
        Counter({('a','b'): 1, ('b','c'): 1})
    """
    chars = list(text)
    ngrams = [tuple(chars[i: i + n]) for i in range(len(chars) - n + 1)]
    return Counter(ngrams)


def chrf_sentence(hypothesis, reference, max_n=6, beta=2):
    """
    Compute the chrF score for a single sentence pair.

    chrF is the F-score that combines:
      - Precision: what fraction of hypothesis n-grams appear in the reference?
      - Recall: what fraction of reference n-grams appear in the hypothesis?

    F-score = (1 + beta²) * precision * recall / (beta² * precision + recall)

    beta=2 gives more weight to recall (the standard setting for MT evaluation).

    Args:
        hypothesis (str): The model's translation.
        reference (str): The human reference translation.
        max_n (int): Maximum character n-gram size. Default: 6.
        beta (float): Weight for recall vs precision. Default: 2.

    Returns:
        float: chrF score from 0 to 100.
    """
    if not hypothesis or not reference:
        return 0.0

    total_precision = 0.0
    total_recall = 0.0
    count = 0

    for n in range(1, max_n + 1):
        hyp_ngrams = get_char_ngrams(hypothesis, n)
        ref_ngrams = get_char_ngrams(reference, n)

        if not hyp_ngrams or not ref_ngrams:
            continue

        # Count matching n-grams (clipped by reference count)
        matching = sum(
            min(count, ref_ngrams.get(ngram, 0))
            for ngram, count in hyp_ngrams.items()
        )

        precision = matching / sum(hyp_ngrams.values())
        recall = matching / sum(ref_ngrams.values())

        total_precision += precision
        total_recall += recall
        count += 1

    if count == 0:
        return 0.0

    # Average precision and recall across all n-gram sizes
    avg_precision = total_precision / count
    avg_recall = total_recall / count

    if avg_precision + avg_recall == 0:
        return 0.0

    # F-score with beta weighting
    beta_sq = beta ** 2
    chrf = (1 + beta_sq) * avg_precision * avg_recall / (beta_sq * avg_precision + avg_recall)
    return round(chrf * 100, 2)


def corpus_chrf(hypotheses, references, max_n=6, beta=2):
    """
    Compute the average chrF score over a full test set.

    Args:
        hypotheses (list of str): All model translations.
        references (list of str): All reference translations.
        max_n (int): Maximum character n-gram size.
        beta (float): Weight for recall.

    Returns:
        float: Average chrF score from 0 to 100.
    """
    assert len(hypotheses) == len(references)

    if not hypotheses:
        return 0.0

    scores = [
        chrf_sentence(hyp, ref, max_n=max_n, beta=beta)
        for hyp, ref in zip(hypotheses, references)
    ]

    return round(sum(scores) / len(scores), 2)
