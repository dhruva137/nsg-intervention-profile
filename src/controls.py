"""Controls that separate real decision information in an explanation from generic
scaffolding: explanation-swap, shuffled-explanation, and length-matched-null.

The idea, shared across faithfulness work this repo cites (e.g. the CoT-unfaithfulness
literature): if a predictor's simulatability accuracy survives an intervention that
should destroy the explanation's actual content while preserving its surface form (same
length, same vocabulary, same "there is an explanation here" cue), then whatever NSG that
condition earns is not coming from the explanation being informative -- it is scaffolding.

One honest note up front, expanded on in the README: this repo's explanations are fixed
templates ("The decision was based on X.") rather than row-specific natural language, per
the exact wording specified for this experiment. `explanation_swap` on a fixed template is
consequently a no-op -- swapping in another row's copy of an identical string changes
nothing. `shuffle_explanation` and `length_matched_null` do not have this problem because
they act on the string itself rather than exchanging it for another instance of the same
string, and they are the controls this repo leans on for the "scaffolding, not
information" claim.
"""
from __future__ import annotations

import random
from typing import Sequence

_NULL_FILLER = (
    "the decision process reviewed several relevant considerations before "
    "reaching its conclusion in this particular case"
)


def explanation_swap(
    explanations: Sequence[str],
    rng: random.Random | None = None,
) -> list[str]:
    """Give each item a different item's explanation, matched by closest length.

    Holds the input, the reference model's answer, and the label fixed; only the
    explanation text changes, and it changes to text that was not actually generated for
    this example. If a predictor's accuracy is unaffected by this swap, it is not using
    example-specific content from the explanation.

    Implementation note: sort by length once (O(n log n)) and pair each item with its
    neighbour in that order, rather than re-searching all remaining candidates for every
    item (an earlier version of this function did that and was O(n^2 log n) -- fine for a
    handful of rows, but proxy_gaming.py calls this on eval sets in the thousands, where
    the quadratic version alone blew the script's 60-second budget).
    """
    rng = rng or random.Random(0)
    n = len(explanations)
    if n <= 1:
        return list(explanations)

    # Tie-break with a random key so items of equal length (the common case here, since
    # this repo's explanations are fixed templates) still get shuffled rather than paired
    # in input order every time.
    order = sorted(range(n), key=lambda i: (len(explanations[i]), rng.random()))
    swapped: list[str] = [""] * n
    for pos, i in enumerate(order):
        j = order[(pos + 1) % n]  # neighbour in length-sorted order = closest length, i != j
        swapped[i] = explanations[j]
    return swapped


def shuffle_explanation(explanation: str, rng: random.Random | None = None) -> str:
    """Randomly reorder the words of one explanation.

    Same tokens, same length, randomised order: this destroys phrase structure (e.g. the
    contiguous phrase "based on X" that `nsg.parse_named_feature` looks for) while
    preserving vocabulary and length, so any drop in NSG cannot be blamed on the control
    removing information a length- or vocabulary-matched null would also lack.
    """
    rng = rng or random.Random(0)
    stripped = explanation.rstrip(".")
    words = stripped.split()
    if len(words) <= 1:
        return explanation
    shuffled = words[:]
    rng.shuffle(shuffled)
    return " ".join(shuffled) + "."


def length_matched_null(explanation: str) -> str:
    """A generic explanation of approximately equal character length that names no
    feature at all -- the floor NSG should sit at if the predictor is genuinely reading
    the explanation's content rather than just noticing something is there.
    """
    target_len = len(explanation)
    if target_len == 0:
        return ""
    filler_words = _NULL_FILLER.split()
    out = ""
    i = 0
    while len(out) < target_len:
        out += filler_words[i % len(filler_words)] + " "
        i += 1
    return out[:target_len].rstrip() + "."
