# lalango/evaluation/report.py
#
# Generates a human-readable evaluation report after running the model on a test set.

from lalango.evaluation.bleu import corpus_bleu
from lalango.evaluation.chrf import corpus_chrf


def generate_report(hypotheses, references, lang_pair, model_name, num_examples=5):
    """
    Generate and print a full evaluation report.

    Args:
        hypotheses (list of str): The model's translations for the test set.
        references (list of str): The human reference translations.
        lang_pair (str): e.g. "konkani-english"
        model_name (str): e.g. "seq2seq_lstm"
        num_examples (int): How many example translations to show. Default: 5.

    Returns:
        dict: The computed metrics (useful for saving results to a file).
    """
    assert len(hypotheses) == len(references), (
        "Number of hypotheses and references must match."
    )

    # Compute metrics
    bleu = corpus_bleu(hypotheses, references)
    chrf = corpus_chrf(hypotheses, references)

    # Print the report
    print("=" * 60)
    print("  La Lango AI — Evaluation Report")
    print("=" * 60)
    print(f"  Language pair : {lang_pair}")
    print(f"  Model         : {model_name}")
    print(f"  Test set size : {len(hypotheses)} sentences")
    print("-" * 60)
    print(f"  BLEU  : {bleu:.1f}")
    print(f"  chrF  : {chrf:.1f}")
    print("-" * 60)

    # Show a sample of translations so we can sanity-check the output
    print(f"\n  Example translations (first {num_examples}):\n")
    for i in range(min(num_examples, len(hypotheses))):
        # Find the source — we only have hypothesis and reference here,
        # so we show both and let the reader compare
        print(f"  [{i+1}]")
        print(f"    Reference : {references[i]}")
        print(f"    Predicted : {hypotheses[i]}")
        print()

    print("=" * 60)

    return {
        "lang_pair": lang_pair,
        "model": model_name,
        "test_size": len(hypotheses),
        "bleu": bleu,
        "chrf": chrf,
    }
