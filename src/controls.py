"""Explanation-swap control for NSG intervention profile.

The explanation-swap control holds the answer fixed and substitutes an
explanation from a *different* example of similar length and style.  This
separates genuine decision information from generic explanation scaffolding
and answer leakage.

If NSG stays high under explanation-swap, the predictor is reading the answer
or the format, not the decision content of the explanation.
"""
from __future__ import annotations
import random


def explanation_swap(
    explanations: list[str],
    rng: random.Random | None = None,
    length_tolerance: float = 0.3,
) -> list[str]:
    """Return a permuted explanation list where each explanation comes from a different example.

    Matching is by length similarity: each explanation is replaced by the
    explanation whose length is closest to its own, subject to the constraint
    that no explanation is matched to itself.

    length_tolerance: acceptable relative length difference.  If no match is
    found within tolerance, fall back to the next closest.
    """
    rng = rng or random.Random(0)
    n = len(explanations)
    lengths = [len(e) for e in explanations]
    swapped = [None] * n
    used = set()

    indices = list(range(n))
    rng.shuffle(indices)  # randomise matching order to avoid systematic bias

    for i in indices:
        target_len = lengths[i]
        # Find best match: different index, not yet used, closest length
        candidates = sorted(
            [j for j in range(n) if j != i and j not in used],
            key=lambda j: abs(lengths[j] - target_len),
        )
        if candidates:
            best = candidates[0]
            swapped[i] = explanations[best]
            used.add(best)
        else:
            # Last resort: use any remaining
            remaining = [j for j in range(n) if j not in used]
            if remaining:
                best = remaining[0]
                swapped[i] = explanations[best]
                used.add(best)
            else:
                swapped[i] = explanations[(i + 1) % n]  # fallback

    return swapped  # type: ignore[return-value]


def length_match_explanations(
    explanations: list[str],
    style_template: str = "",
) -> list[str]:
    """Pad or truncate explanations to match the median length.

    Used when comparing across conditions with different typical lengths,
    to control for length as a confound in the leakage probe.
    """
    if not explanations:
        return []
    target = int(sorted([len(e) for e in explanations])[len(explanations) // 2])
    result = []
    for e in explanations:
        if len(e) >= target:
            result.append(e[:target])
        else:
            result.append(e + " " + style_template[:target - len(e) - 1])
    return result