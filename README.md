# nsg-intervention-profile

An argument that Normalised Simulatability Gain can be gamed by an explanation that names a
correlated proxy instead of the real cause, and a proposal to report three numbers instead of one.

Everything here is synthetic and CPU-only. The LLM version is written but has not been run.

## The claim I'm looking at

Mayne et al., *A Positive Case for Faithfulness* (https://arxiv.org/abs/2602.02639), define NSG:
a predictor tries to guess a reference model's answer on a nearby counterfactual, with and without
that model's explanation, and the gain is normalised against the headroom above the no-explanation
baseline.

```
NSG = (acc_with - acc_without) / (1 - acc_without)
```

Self-explanations score consistently positive, beating cross-model explanations. The paper reads
this as evidence for faithfulness.

## What I want to test

NSG measures whether an explanation *helps you predict*. That isn't the same as whether it names
the real cause.

If a model decides on feature A but its explanation names feature B, and A and B are correlated in
the data the counterfactuals come from, then "it was based on B" still predicts well. The
explanation earns a good NSG score while being causally wrong.

So: does that advantage survive if you deliberately break the A–B correlation?

This extends a limitation the paper already flags — it doesn't contradict it. Mayne et al. name
counterfactual distance and counterfactual-region quality as open problems. My point is that
*which* counterfactuals you sample isn't a protocol detail, it decides whether the metric can tell a
decision rule from a correlate at all.

## The intervention profile

Instead of one averaged NSG, report three:

1. **Ordinary** — counterfactuals from the natural distribution, where A and B move together
2. **Decorrelated** — counterfactuals built so A and B disagree
3. **Targeted** — interventions on whatever feature the explanation actually named

The spread between them is the diagnostic. The average hides it.

## What I actually found, and what's just algebra

I need to be straight about this, because it's the main limitation of the repo.

The headline numbers are **analytically derivable, not empirical discoveries.** In my setup the
reference model is a rule using only `true_feature`, and `proxy_feature` agrees with it at rate ρ.
It falls out that:

- `proxy_ordinary` = **2ρ − 1**, near exactly (0.202 at ρ=0.6, 0.401 at 0.7, 0.980 at 0.99)
- `proxy_decorrelated` = **−1**, because on the subset where the two disagree, an explanation naming
  the proxy is wrong every single time by definition

So the simulation confirms arithmetic. It did not discover anything. I'm keeping it because it makes
the averaging problem concrete and gives the effect a size, but calling it a finding would be
overclaiming and I'd rather say so up front.

The part I think genuinely stands on its own is the *argument*: reporting one averaged NSG is not
enough, and the three-number profile is cheap to compute and would have surfaced this.

### Results

`python experiments/proxy_gaming.py` — 7 ρ values × 20 seeds, ~14s on CPU, 95% bootstrap CIs:

```
  rho        honest_ordinary         proxy_ordinary      honest_decorrelated       proxy_decorrelated
 0.50 +1.000 [+1.000, +1.000] +0.002 [-0.003, +0.008]  +1.000 [+1.000, +1.000]  -0.001 [-0.110, +0.105]
 0.70 +1.000 [+1.000, +1.000] +0.401 [+0.393, +0.409]  +1.000 [+1.000, +1.000]  -1.015 [-1.049, -0.983]
 0.90 +1.000 [+1.000, +1.000] +0.799 [+0.794, +0.804]  +1.000 [+1.000, +1.000]  -0.995 [-1.051, -0.944]
 0.99 +1.000 [+1.000, +1.000] +0.980 [+0.978, +0.982]  +1.000 [+1.000, +1.000]  -1.029 [-1.126, -0.919]
```

At ρ=0.99 the proxy explanation is within 0.02 of the honest one under ordinary counterfactuals —
no single-number report would flag that as a problem — and goes to −1.03 once decorrelated. At
ρ=0.50 it earns nothing, because with no correlation there's nothing to hide behind.

`python experiments/explanation_controls.py` — the full profile plus controls at ρ=0.9:

```
 condition             ordinary         decorrelated             targeted
    honest +1.000 [+1.000, +1.000] +1.000 [+1.000, +1.000] +1.000 [+1.000, +1.000]
     proxy +0.801 [+0.794, +0.807] -1.032 [-1.115, -0.956] -0.804 [-0.824, -0.780]

 condition                 real                 swap             shuffled                 null
    honest +1.000 [+1.000, +1.000] +1.000 [+1.000, +1.000] +0.037 [+0.020, +0.054] +0.002 [-0.013, +0.017]
     proxy +0.801 [+0.794, +0.807] +0.801 [+0.794, +0.807] +0.031 [+0.014, +0.048] +0.002 [-0.013, +0.017]
```

Honest reads flat across all three regimes. Proxy reads +0.80 / −1.03 / −0.80. The spread is the
signal; the average of those three is meaningless.

Shuffled and null controls both collapse to ≤ +0.04, so what's being measured is explanation
content rather than scaffolding.

**One control failed to do anything, and I'm reporting it rather than dropping it.** The
explanation-swap column is identical to the real column for both conditions. That's because my
explanations are fixed templates ("The decision was based on X"), the same string for every row in a
condition — so swapping one row's explanation for another's changes nothing. The swap control is
load-bearing on real generations where explanations vary per example. Here it's inert by
construction and tells you nothing.

`python experiments/counterfactual_distance.py` — how often a counterfactual at Hamming distance d
over k features touches the decision-relevant feature. This one is just **p = d/k**; the Monte Carlo
only checks my implementation. It's here because it explains why the paper's Income example can show
positive average NSG while ignoring race and marital status: at k=20, d=1, only 5% of sampled
counterfactuals touch any given feature, so ignoring one costs almost nothing on average.

## Scope — what this is not

- The reference model is a decision rule, not a language model. Ground truth is exact because I
  built it that way, which is what makes it clean and also what makes it artificial.
- ρ is a knob I set. On real data it's a measured property.
- The proxy explanation is imposed by me. Whether models produce proxy explanations on their own is
  a different question and I haven't touched it.
- Nothing here shows real LLM self-explanations have this problem. It shows the problem is possible
  and how big it gets.

## Structure

```
src/           synthetic data, NSG implementation, controls, plotting
experiments/   the three CPU experiments — these run and produce the tables above
notebooks/     llm_extension_t4.ipynb — Kaggle T4, NOT RUN
results/       figures
```

## Running it

```bash
pip install -r requirements.txt
python experiments/proxy_gaming.py
```

All scripts take `--seed` and `--dry-run`. `python build_notebooks.py` regenerates and validates the
notebook.

## Status

CPU experiments: done, run, real numbers above.
LLM notebook: written, never executed. Every cell ships empty. No results are claimed for it.

## Reading

- https://arxiv.org/abs/2602.02639 — Mayne et al., the NSG paper this responds to
- https://arxiv.org/abs/2410.13787 — Binder et al., *Looking Inward*
- https://aclanthology.org/2024.acl-short.49/ — correlational counterfactual testing
- https://aclanthology.org/2025.emnlp-main.529/ — activation-based causal analysis
- https://arxiv.org/abs/2602.20710 — Counterfactual Simulation Training
- https://arxiv.org/abs/2606.18327 — Self-CTRL
- https://aclanthology.org/2025.emnlp-main.504/ — CoT faithfulness by unlearning steps; attacks
  faithfulness by removal rather than prediction, so it isn't vulnerable to the failure above
