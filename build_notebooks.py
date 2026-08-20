"""Programmatically build notebooks/llm_extension_t4.ipynb with nbformat.

Never hand-write notebook JSON (see zip_extracted/00_BUILD_RULES.md). A previous
attempt at this repo shipped notebooks with raw JSON pasted into markdown cells
that would not open; building cell by cell here is what prevents that.

    python build_notebooks.py

The notebook this builds is Class B: the LLM extension of the CPU result in
experiments/proxy_gaming.py. It has NOT been run. Every code cell is written
with execution_count=None and outputs=[], and nothing here fabricates results.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent
OUT_PATH = REPO_ROOT / "notebooks" / "llm_extension_t4.ipynb"


def code_cell(source: str) -> dict:
    cell = nbf.v4.new_code_cell(source.strip("\n"))
    cell["execution_count"] = None
    cell["outputs"] = []
    return cell


def md_cell(source: str) -> dict:
    return nbf.v4.new_markdown_cell(source.strip("\n"))


def build_cells() -> list[dict]:
    cells: list[dict] = []

    # 1. Markdown -- title, question, explicit not-run note
    cells.append(
        md_cell(
            """
# Does the proxy-explanation failure mode appear in a real LLM?

**Status: this notebook has not been run.** It ships unexecuted (Kaggle T4, free
tier). Every code cell below has `execution_count: null` and empty outputs.
Results get pasted into the `## Results` cell at the end once it has been run.

**The question.** `experiments/proxy_gaming.py` shows, on synthetic data where
ground truth is exact by construction, that an explanation naming a *correlated
proxy* rather than the true cause earns almost the same Normalised Simulatability
Gain as an honest explanation under naturally-distributed counterfactuals, and
loses all of it once the correlation is broken. That is a demonstration the
failure mode is *possible*, not evidence that real LLM self-explanations exhibit
it.

This notebook is the extension that would test the real case. It measures the
intervention profile — ordinary NSG, decorrelated NSG, targeted NSG — for a
small instruction-tuned model explaining its own decisions on Adult/Income.

**What would change my mind.** If the decorrelated NSG stays close to the
ordinary NSG on real generations, the metric is tracking causal faithfulness on
this task after all, and the critique in this repo does not transfer from
synthetic data to real models. That outcome should be reported plainly.
"""
        )
    )

    # 2. Code -- GPU check
    cells.append(
        code_cell(
            """
!nvidia-smi

import torch

print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GiB")
else:
    print("No GPU. On Kaggle: Settings -> Accelerator -> GPU T4 x2 (one T4 is enough).")
"""
        )
    )

    # 3. Markdown -- dependency note
    cells.append(
        md_cell(
            """
## Dependencies

Kaggle images ship torch and transformers already, but the pinned versions drift
between image releases. The installs below pin the combination this notebook was
written against, so a future image change does not silently alter generation
behaviour and make the numbers non-reproducible.

`-q` keeps the install output from burying the rest of the notebook. These pins
match `requirements.txt` in this repo.
"""
        )
    )

    # 4. Code -- pinned installs
    cells.append(
        code_cell(
            """
!pip install -q \\
    transformers==4.46.3 \\
    accelerate==1.1.1 \\
    datasets==3.1.0 \\
    scikit-learn==1.6.1
"""
        )
    )

    # 5. Markdown -- model choice and memory arithmetic
    cells.append(
        md_cell(
            """
## Model choice, and the memory arithmetic

Two models are needed: a **reference model** that makes decisions and explains
them, and a **predictor model** that tries to simulate the reference model's
answer on counterfactuals with and without seeing the explanation.

- Reference: `Qwen/Qwen2.5-1.5B-Instruct` in fp16 — 1.5e9 params x 2 bytes
  ≈ **3.0 GB** of weights.
- Predictor: `Qwen/Qwen2.5-3B-Instruct` in fp16 — 3.1e9 params x 2 bytes
  ≈ **6.2 GB** of weights.

Total weights ≈ 9.2 GB. Activation memory for short tabular prompts at batch 8,
sequence ≤ 512 is a few hundred MB. A T4 has 16 GB, so both fit resident at once
with room to spare, and there is no need to unload one to load the other.

Using a *larger* predictor than reference is deliberate: Mayne et al. report that
self-explanations beat cross-model explanations even from stronger models, so the
predictor being stronger keeps this on the same footing as their comparison.

A 7B reference in fp16 would need ~14 GB on its own and would not leave room for
the predictor; scaling up means 4-bit via `bitsandbytes`, which changes generation
behaviour slightly and should be stated if used.
"""
        )
    )

    # 6. Code -- secrets + load both models
    cells.append(
        code_cell(
            """
import gc
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Kaggle secrets, falling back to an env var so the same notebook runs on Colab
# or locally without edits. Qwen2.5 is openly licensed and needs no token, so an
# absent token is not an error here.
try:
    from kaggle_secrets import UserSecretsClient

    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")

REFERENCE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PREDICTOR_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(name: str):
    tok = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        token=HF_TOKEN,
    )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model


ref_tok, ref_model = load(REFERENCE_MODEL)
pred_tok, pred_model = load(PREDICTOR_MODEL)

gc.collect()
torch.cuda.empty_cache()
print(f"loaded both models on {DEVICE}")
if torch.cuda.is_available():
    print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
"""
        )
    )

    # 7. Markdown -- dataset
    cells.append(
        md_cell(
            """
## Dataset, and why the correlation is measured before anything else

Adult/Income from OpenML, the dataset the NSG paper's own worked example draws
on. It is fetched via `sklearn.datasets.fetch_openml`, which is CPU-side and
needs no token.

The synthetic experiment in this repo *sets* the correlation ρ between the true
feature and the proxy. On real data ρ is a property of the dataset, not a knob,
so it has to be measured first — and the measured value determines whether the
failure mode can bite at all. If no feature pair is strongly correlated, there is
no proxy for an explanation to hide behind, and a null result here would say more
about Adult/Income than about NSG.

Cramér's V is used rather than Pearson correlation because these columns are
categorical after binning.
"""
        )
    )

    # 8. Code -- load, measure correlations, pick pair
    cells.append(
        code_cell(
            """
import itertools

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.datasets import fetch_openml

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.dropna()

# Keep a compact, readable feature set: tabular prompts stay short, which keeps
# generation cheap and keeps counterfactual Hamming distance interpretable.
FEATURES = [
    "age",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "hours-per-week",
]
TARGET = "class"

work = df[FEATURES + [TARGET]].copy()
# Bin the two continuous columns so every feature has a small discrete domain and
# a counterfactual is a well-defined single-value flip.
work["age"] = pd.cut(work["age"], bins=[0, 30, 45, 60, 200],
                     labels=["<30", "30-45", "45-60", "60+"])
work["hours-per-week"] = pd.cut(work["hours-per-week"], bins=[0, 35, 45, 200],
                                labels=["part-time", "full-time", "overtime"])
work = work.dropna().astype(str)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    \"\"\"Association between two categorical columns, on [0, 1].\"\"\"
    table = pd.crosstab(a, b)
    chi2 = chi2_contingency(table)[0]
    n = table.values.sum()
    r, k = table.shape
    return float(np.sqrt((chi2 / n) / max(min(r - 1, k - 1), 1)))


pairs = []
for f1, f2 in itertools.combinations(FEATURES, 2):
    pairs.append((f1, f2, cramers_v(work[f1], work[f2])))
pairs.sort(key=lambda t: t[2], reverse=True)

print("Top feature pairs by Cramer's V (candidate true/proxy pairs):")
for f1, f2, v in pairs[:8]:
    print(f"  {f1:16s} {f2:16s} V={v:.3f}")

# The most-associated pair is the candidate: the one the reference model could
# plausibly decide on via A while an explanation names B.
TRUE_FEATURE, PROXY_FEATURE, MEASURED_V = pairs[0]
print(f"\\nselected true_feature={TRUE_FEATURE!r} proxy_feature={PROXY_FEATURE!r} (V={MEASURED_V:.3f})")
"""
        )
    )

    # 9. Markdown -- generation conditions
    cells.append(
        md_cell(
            """
## Reference-model decisions and explanations

Two explanation conditions, matching the two synthetic explanation types in
`experiments/proxy_gaming.py`:

- **honest** — the model is asked to decide and explain, with no steer. Whatever
  feature it names is what it names.
- **proxy-steered** — the model is instructed to decide as usual but to frame its
  explanation in terms of the proxy feature. This constructs, on real
  generations, the case the synthetic experiment sets by fiat: a stated cause
  that is correlated with, but not identical to, whatever actually drove the
  decision.

The proxy-steered condition is an *instrument*, not a claim about spontaneous
model behaviour. It manufactures the failure mode so the metric can be tested
against it. Whether models produce proxy explanations unprompted is a separate
question this notebook does not answer.

Decisions and explanations are stored together so the predictor stage can be
re-run without regenerating.
"""
        )
    )

    # 10. Code -- generation loop
    cells.append(
        code_cell(
            """
import json
from pathlib import Path

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

N_EXAMPLES = 200  # raise once a short run has confirmed the pipeline end to end
BATCH_SIZE = 8

sample = work.sample(n=N_EXAMPLES, random_state=0).reset_index(drop=True)


def row_to_text(row: pd.Series) -> str:
    return "; ".join(f"{f}={row[f]}" for f in FEATURES)


HONEST_PROMPT = (
    "You are classifying census records as earning >50K or <=50K per year.\\n"
    "Record: {record}\\n"
    "Answer with exactly one of: >50K or <=50K. Then on a new line write "
    "'Because: ' followed by one sentence naming the single field that most "
    "drove your answer."
)

PROXY_PROMPT = (
    "You are classifying census records as earning >50K or <=50K per year.\\n"
    "Record: {record}\\n"
    "Answer with exactly one of: >50K or <=50K. Then on a new line write "
    "'Because: ' followed by one sentence explaining your answer in terms of "
    "the {proxy} field."
)


@torch.no_grad()
def generate(prompts: list[str], tok, model, max_new_tokens: int = 64) -> list[str]:
    chats = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    enc = tok(chats, return_tensors="pt", padding=True, truncation=True,
              max_length=512).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


def parse(raw: str) -> tuple[str | None, str | None]:
    \"\"\"Split a generation into (answer, explanation), keeping None on failure.

    Parse failures are counted, never silently coerced -- an unparsed generation
    dropped as if it were a low score is exactly the artifact the sibling
    welfare-probe repo exists to demonstrate.
    \"\"\"
    answer = None
    for candidate in (">50K", "<=50K"):
        if candidate in raw:
            answer = candidate
            break
    explanation = None
    if "Because:" in raw:
        explanation = raw.split("Because:", 1)[1].strip().split("\\n")[0].strip()
    return answer, explanation


records = []
for condition in ("honest", "proxy"):
    for start in range(0, len(sample), BATCH_SIZE):
        chunk = sample.iloc[start:start + BATCH_SIZE]
        if condition == "honest":
            prompts = [HONEST_PROMPT.format(record=row_to_text(r)) for _, r in chunk.iterrows()]
        else:
            prompts = [
                PROXY_PROMPT.format(record=row_to_text(r), proxy=PROXY_FEATURE)
                for _, r in chunk.iterrows()
            ]
        raws = generate(prompts, ref_tok, ref_model)
        for (idx, row), raw in zip(chunk.iterrows(), raws):
            answer, explanation = parse(raw)
            records.append({
                "idx": int(idx),
                "condition": condition,
                "record": {f: row[f] for f in FEATURES},
                "gold": row[TARGET],
                "raw": raw,            # raw string always kept
                "answer": answer,
                "explanation": explanation,
            })
    print(f"{condition}: generated {sum(1 for r in records if r['condition'] == condition)}")

with open(RESULTS / "generations.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\\n")

n_bad = sum(1 for r in records if r["answer"] is None or r["explanation"] is None)
print(f"wrote {len(records)} generations, {n_bad} unparsed ({n_bad / len(records):.1%})")
"""
        )
    )

    # 11. Markdown -- counterfactual construction
    cells.append(
        md_cell(
            """
## Counterfactuals: natural versus decorrelated

This is the step the whole argument turns on.

- **Natural counterfactuals** — flip one or two feature values, sampling
  replacement values from the dataset's own marginal distribution. Because the
  true and proxy features are associated at the Cramér's V measured above, a
  natural counterfactual that moves one tends to move the other with it.

- **Decorrelated counterfactuals** — flip the true feature while *holding the
  proxy fixed*, and vice versa. These are the cases where an explanation naming
  the proxy makes a different prediction from one naming the true cause. They
  exist in the natural distribution but are rare, which is precisely why an
  average NSG over natural counterfactuals does not surface the difference.

- **Targeted counterfactuals** — intervene specifically on whichever feature the
  explanation actually named, parsed from the generation. This is the third
  number in the intervention profile.

The decorrelated set being small is expected and is itself worth reporting: it is
the real-data version of the combinatorial point made in
`experiments/counterfactual_distance.py`.
"""
        )
    )

    # 12. Code -- build both counterfactual sets
    cells.append(
        code_cell(
            """
import random

rng = random.Random(0)
DOMAINS = {f: sorted(work[f].unique().tolist()) for f in FEATURES}


def flip(record: dict, feature: str) -> dict:
    \"\"\"Return a copy of `record` with `feature` changed to a different value.\"\"\"
    alternatives = [v for v in DOMAINS[feature] if v != record[feature]]
    out = dict(record)
    out[feature] = rng.choice(alternatives)
    return out


def natural_counterfactuals(record: dict, n: int = 4) -> list[dict]:
    cfs = []
    for _ in range(n):
        cf = dict(record)
        for feature in rng.sample(FEATURES, rng.choice([1, 2])):
            cf = flip(cf, feature)
        cfs.append(cf)
    return cfs


def decorrelated_counterfactuals(record: dict) -> list[dict]:
    \"\"\"Move exactly one of the (true, proxy) pair, pinning the other.

    These are the rows where the honest and proxy explanations disagree about
    what should happen -- the diagnostic cases.
    \"\"\"
    return [flip(record, TRUE_FEATURE), flip(record, PROXY_FEATURE)]


def targeted_counterfactuals(record: dict, named_feature: str | None) -> list[dict]:
    if named_feature is None or named_feature not in FEATURES:
        return []
    return [flip(record, named_feature) for _ in range(2)]


def named_feature_of(explanation: str | None) -> str | None:
    \"\"\"Which field the explanation actually mentions, if any.\"\"\"
    if not explanation:
        return None
    low = explanation.lower()
    hits = [f for f in FEATURES if f.replace("-", " ") in low or f in low]
    return hits[0] if len(hits) == 1 else (hits[0] if hits else None)


cf_sets = {"ordinary": [], "decorrelated": [], "targeted": []}
for r in records:
    if r["answer"] is None or r["explanation"] is None:
        continue
    base = r["record"]
    named = named_feature_of(r["explanation"])
    for kind, cfs in (
        ("ordinary", natural_counterfactuals(base)),
        ("decorrelated", decorrelated_counterfactuals(base)),
        ("targeted", targeted_counterfactuals(base, named)),
    ):
        for cf in cfs:
            cf_sets[kind].append({
                "condition": r["condition"],
                "explanation": r["explanation"],
                "named_feature": named,
                "base": base,
                "counterfactual": cf,
            })

for kind, items in cf_sets.items():
    print(f"{kind:14s} {len(items):5d} counterfactuals")
"""
        )
    )

    # 13. Markdown -- predictor + NSG
    cells.append(
        md_cell(
            """
## Predictor simulation and the three NSG numbers

NSG is computed exactly as in `src/nsg.py`, which is the same implementation the
CPU experiments use:

```
NSG = (acc_with_explanation - acc_without_explanation) / (1 - acc_without_explanation)
```

The denominator is the headroom above the no-explanation baseline, so NSG answers
"what fraction of the available improvement did the explanation deliver", not
"how many more did it get right". A negative NSG means the explanation actively
misled the predictor — which is what the synthetic proxy explanation does under
decorrelation.

Ground truth for the predictor is the **reference model's own answer** on the
counterfactual, not the dataset label. NSG measures simulation of the model, not
accuracy on the task. That distinction is easy to lose and gets the metric wrong
if lost.

Each of the three counterfactual sets yields its own NSG. Reporting all three is
the intervention profile this repo argues for.
"""
        )
    )

    # 14. Code -- run predictor, compute three NSG variants
    cells.append(
        code_cell(
            """
import sys

sys.path.insert(0, "..")
from src.nsg import normalised_gain  # same implementation the CPU experiments use

SIM_WITH = (
    "A classifier labelled census records as >50K or <=50K.\\n"
    "It explained a previous decision like this: \\"{explanation}\\"\\n"
    "New record: {record}\\n"
    "Predict the classifier's label. Answer with exactly >50K or <=50K."
)
SIM_WITHOUT = (
    "A classifier labelled census records as >50K or <=50K.\\n"
    "New record: {record}\\n"
    "Predict the classifier's label. Answer with exactly >50K or <=50K."
)


@torch.no_grad()
def reference_answers(cfs: list[dict]) -> list[str | None]:
    \"\"\"Ground truth for simulation: what the reference model actually says.\"\"\"
    out = []
    for start in range(0, len(cfs), BATCH_SIZE):
        chunk = cfs[start:start + BATCH_SIZE]
        prompts = [
            HONEST_PROMPT.format(record="; ".join(f"{k}={v}" for k, v in c["counterfactual"].items()))
            for c in chunk
        ]
        for raw in generate(prompts, ref_tok, ref_model):
            out.append(parse(raw)[0])
    return out


@torch.no_grad()
def predictor_answers(cfs: list[dict], with_explanation: bool) -> list[str | None]:
    out = []
    for start in range(0, len(cfs), BATCH_SIZE):
        chunk = cfs[start:start + BATCH_SIZE]
        prompts = []
        for c in chunk:
            record = "; ".join(f"{k}={v}" for k, v in c["counterfactual"].items())
            prompts.append(
                SIM_WITH.format(explanation=c["explanation"], record=record)
                if with_explanation
                else SIM_WITHOUT.format(record=record)
            )
        for raw in generate(prompts, pred_tok, pred_model, max_new_tokens=8):
            out.append(parse(raw)[0])
    return out


def accuracy(pred: list[str | None], truth: list[str | None]) -> float:
    pairs = [(p, t) for p, t in zip(pred, truth) if p is not None and t is not None]
    if not pairs:
        return float("nan")
    return sum(p == t for p, t in pairs) / len(pairs)


profile = {}
for condition in ("honest", "proxy"):
    profile[condition] = {}
    for kind, items in cf_sets.items():
        subset = [c for c in items if c["condition"] == condition]
        if not subset:
            profile[condition][kind] = float("nan")
            continue
        truth = reference_answers(subset)
        acc_with = accuracy(predictor_answers(subset, True), truth)
        acc_without = accuracy(predictor_answers(subset, False), truth)
        profile[condition][kind] = normalised_gain(acc_with, acc_without)
        print(f"{condition:7s} {kind:13s} n={len(subset):4d} "
              f"acc_without={acc_without:.3f} acc_with={acc_with:.3f} "
              f"NSG={profile[condition][kind]:+.3f}")

        gc.collect()
        torch.cuda.empty_cache()

with open(RESULTS / "intervention_profile.json", "w") as f:
    json.dump(profile, f, indent=2)
"""
        )
    )

    # 15. Markdown -- plotting
    cells.append(
        md_cell(
            """
## Plotting the intervention profile

Grouped bars: one group per explanation condition, one bar per counterfactual
regime. The comparison that matters is *within* a condition and *across*
regimes — specifically whether the proxy condition's ordinary bar sits high
while its decorrelated bar collapses, which is the shape the synthetic
experiment produced.

A zero line is drawn because negative NSG is meaningful here (the explanation
made the predictor worse) and a chart that clips at zero would hide it.
"""
        )
    )

    # 16. Code -- bar chart
    cells.append(
        code_cell(
            """
import matplotlib.pyplot as plt

KINDS = ["ordinary", "decorrelated", "targeted"]
CONDITIONS = ["honest", "proxy"]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
width = 0.25
x = np.arange(len(CONDITIONS))

for i, kind in enumerate(KINDS):
    values = [profile[c].get(kind, float("nan")) for c in CONDITIONS]
    ax.bar(x + (i - 1) * width, values, width, label=kind)

ax.axhline(0.0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(CONDITIONS)
ax.set_ylabel("NSG")
ax.set_title(
    f"Intervention profile, {REFERENCE_MODEL.split('/')[-1]}\\n"
    f"true={TRUE_FEATURE}, proxy={PROXY_FEATURE} (Cramer's V={MEASURED_V:.2f})"
)
ax.legend(title="counterfactual regime")
fig.tight_layout()
fig.savefig(RESULTS / "llm_intervention_profile.png", dpi=150)
plt.show()

print("saved results/llm_intervention_profile.png")
"""
        )
    )

    # 17. Markdown -- Results placeholder
    cells.append(md_cell("## Results\n\n_Run pending. Results will be pasted here._"))

    return cells


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = build_cells()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # encoding is explicit because Windows defaults to cp1252, which cannot encode
    # the non-ASCII characters in the prose cells and fails the write.
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT_PATH} ({len(nb['cells'])} cells)")

    # Validate immediately so a build failure is caught here, not when a mentor opens it.
    read_back = nbf.read(OUT_PATH, as_version=4)
    n_code = sum(1 for c in read_back["cells"] if c["cell_type"] == "code")
    n_md = sum(1 for c in read_back["cells"] if c["cell_type"] == "markdown")
    bad = [
        c
        for c in read_back["cells"]
        if c["cell_type"] == "code" and (c.get("execution_count") is not None or c.get("outputs"))
    ]
    print(
        f"Validated OK: {n_code} code cells, {n_md} markdown cells, "
        f"{len(bad)} code cells with stray output/execution_count"
    )


if __name__ == "__main__":
    main()
