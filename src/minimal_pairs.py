"""Build the leak-free minimal-pair test set from the ORIGINAL agr_50 file.

For each held-out test row we form a minimal pair:
    prefix          = orig_sentence tokens up to (but not including) the verb
    correct_verb    = the original verb (grammatical; original corpus is all-grammatical)
    incorrect_verb  = flip_verb_agreement_perfect(correct_verb)   # ungrammatical

verb_index in the data is 1-based; the verb sits at 0-based position verb_index-1
(this matches Kelly's `verb_idx = int(row['verb_index']) - 1`).

has_attractor = (n_diff_intervening > 0): an intervening noun of opposite number.

We KEEP only pairs where both verb forms are single in-vocab tokens, so the
probability comparison is well defined. Coverage is reported by the caller.
"""
import csv
import sys

from verb_flip import flip_verb_agreement_perfect

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def build_minimal_pairs(original_path, test_indices, vocab=None):
    """Returns a list of dicts: prefix_tokens, correct, incorrect, has_attractor,
    n_diff_intervening, distance. If vocab given, filters to in-vocab verb forms."""
    test_indices = set(test_indices)
    pairs = []
    skipped = {"bad_index": 0, "oov_verb": 0, "flip_failed": 0}
    with open(original_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i not in test_indices:
                continue
            toks = (row.get("orig_sentence") or "").split()
            try:
                vidx = int(row["verb_index"]) - 1
            except (ValueError, KeyError):
                skipped["bad_index"] += 1
                continue
            if not (0 <= vidx < len(toks)):
                skipped["bad_index"] += 1
                continue
            correct = toks[vidx]
            incorrect = flip_verb_agreement_perfect(correct, row.get("verb_pos", ""))
            if incorrect == correct or not incorrect:
                skipped["flip_failed"] += 1
                continue
            if vocab is not None and not (vocab.has(correct) and vocab.has(incorrect)):
                skipped["oov_verb"] += 1
                continue
            try:
                n_diff = int(row.get("n_diff_intervening", 0))
            except ValueError:
                n_diff = 0
            pairs.append({
                "prefix": toks[:vidx],
                "correct": correct,
                "incorrect": incorrect,
                "has_attractor": n_diff > 0,
                "n_diff_intervening": n_diff,
                "distance": row.get("distance", ""),
            })
    return pairs, skipped
