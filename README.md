# NSG Intervention Profile

**Claim being tested.** NSG measures whether an explanation helps an observer predict the model on
nearby counterfactuals. That is a useful operational definition, but it is not the same as
identifying the real causal reason for a decision.

**The failure mode.** A model can use feature A internally while its explanation names feature B,
where A and B are correlated in the natural data distribution. If counterfactuals are sampled from
that same distribution, the proxy explanation still predicts behaviour well and earns a high NSG
while giving the wrong causal account.

**What I want to test.** Does the NSG advantage survive when the true cause and the stated cause
are deliberately decorrelated?

## Design

Synthetic tabular task with a true decision feature and a highly correlated proxy. Then:

1. **Ordinary NSG** on naturally-distributed counterfactuals
2. **Decorrelated NSG** on counterfactuals constructed to break the A–B correlation
3. **Targeted NSG** on interventions to features the explanation explicitly claims matter

The three numbers together are the *intervention profile*. Reporting one average NSG hides which
of these is driving the score.

## Controls

Explanation-swap: hold the answer fixed, replace the explanation with a same-model explanation
from a different example matched on length and style. Separates real decision information from
generic explanation scaffolding and answer leakage.

## What would change my mind

If NSG holds up on decorrelated counterfactuals, the metric genuinely tracks causal faithfulness
and this criticism is wrong.

## Status

Synthetic generator and NSG computation in progress.

## Related

- https://arxiv.org/abs/2602.02639 — the NSG paper
- https://aclanthology.org/2024.acl-short.49/
- https://aclanthology.org/2025.emnlp-main.529/
