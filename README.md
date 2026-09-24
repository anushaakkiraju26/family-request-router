# Family Request Router

Fine-tunes `Qwen/Qwen3-1.7B-Base` with a LoRA adapter so a small, cheap model can route a
parent's request to a family-calendar coordinator agent into one of seven handling labels —
turning a frontier-model reasoning step into a pre-filter that runs locally.

## Result

| Metric | Constrained-choice baseline | Fine-tuned (best pass) |
|---|---|---|
| Routing accuracy, 7 labels, held-out split | 28.4% | **55.6%** (+27.2 pts) |

Trained on 448 examples grown from 30 real, eval-labelled requests taken from the
[family-calendar](https://github.com/anushaakkiraju26/family-calendar) evaluation suite.
55.6% is pass 4's figure and the best so far; the most recent pass 6 sits at 45.6%, having
traded overall accuracy for a targeted +24.7 pts on `fast_path_reject`.

Two classes carry the actual finding. `fast_path_conflict` reached **100% recall on its first
training pass** after being split out of `fast_path_reject`, confirming the split was a real
and learnable distinction. `ambiguous_clarify` stayed at **0% recall across six passes** and
three separate interventions — a richer prompt, general contrastive data, and minimal-pair
contrastive data that varies exactly one fact per triplet. Three failed data/prompt fixes in a
row point away from "needs more data" and toward LoRA rank 8 lacking the capacity for that
particular three-way boundary, which makes rank the next single variable worth changing.

Each pass isolates one variable, and the regressions are recorded alongside the wins — see the
Recap in
[`notebooks/finetune_family_request_router_executed.ipynb`](notebooks/finetune_family_request_router_executed.ipynb)
for the pass-by-pass breakdown.

## Approach

Custom-dataset variant of The Gen Academy's [Fine-Tune a Support Ticket Router](https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router).
Same recipe — `Qwen/Qwen3-1.7B-Base` + LoRA via LLaMA-Factory's LLaMA Board
UI — applied to my own data instead of IT support tickets.

## The labels

A request such as *"Add Leo's soccer practice tomorrow from 4 to 5 PM for family-1"* is routed
to one of the seven labels the coordinator uses to decide how to handle it:

| Label | Downstream action |
|---|---|
| `fast_path_mutate` | Direct create/update/move/delete/restore, pending approval |
| `fast_path_read` | Direct list/show, no mutation |
| `fast_path_reject` | Must be refused by a deterministic safeguard: past-dated event, cross-family access, stale plan version, or an unsafe retry after a rejection |
| `fast_path_conflict` | The same child or parent is double-booked — can be offered alternatives or flagged as a likely error, rather than simply declined |
| `deep_weekly_workflow` | Full specialist pipeline: intake → planner → transportation → reviewer → reminder |
| `outing_workflow` | Constrained outing/activity search wrapper, never mutates the calendar |
| `ambiguous_clarify` | Missing info — must ask before acting |

`fast_path_conflict` was split out of `fast_path_reject` partway through (pass 4) because the
two need different downstream handling; the label count moved from six to seven at that point.

## Files

| File | What it is |
|---|---|
| `notebooks/finetune_family_request_router.ipynb` | The Colab notebook — install, prepare data, train via LLaMA Board, merge, evaluate. Open it directly in Colab. |
| `notebooks/finetune_family_request_router_executed.ipynb` | The same notebook as actually run, outputs included (loss curve, classification reports, confusion matrix, smoke tests) — evidence of the pass 6/7 results documented in the Recap. |
| `tools/generate_dataset.py` | Builds the dataset: 30 real requests (`data/seed_examples.json`) expanded with LLM-generated variations to ~50 balanced examples per label. |
| `data/seed_examples.json` | The 30 real, eval-labelled seed requests this dataset is grounded in — originally from family-calendar's evaluation suite. |
| `data/family_request_routing.csv` | Full generated dataset (`text`, `category_truth`, `source`). |
| `data/family_request_routing_train.csv` / `_val.csv` | Stratified 80/20 split, `text`/`category_truth` only — matches the reference project's CSV shape. |
| `data/family_request_routing_manifest.json` | Label definitions and generation provenance/counts. |

## Regenerating the dataset

Needs an OpenAI-compatible chat model endpoint. Defaults to [Nebius AI Studio](https://studio.nebius.com/);
set `NEBIUS_API_KEY` (and optionally `NEBIUS_BASE_URL`, `--model`) in `.env` — see `.env.example`.

```
pip install -r requirements.txt
python tools/generate_dataset.py --per-class 50
```

## Running the notebook

1. Open `notebooks/finetune_family_request_router.ipynb` in Google Colab (free T4 runtime).
2. Run top to bottom. The dataset-prep cell clones this repo directly, so no manual CSV
   upload is needed as long as it's pushed to GitHub.
3. Everything else — LLaMA Board training, adapter merge, baseline comparison, evaluation —
   follows the reference project's flow; see the notebook's own cells for details.

## Other trial runs

Separate, exploratory fine-tuning runs live alongside the main project's files, each under
a same-named subfolder so they never mix with the primary dataset/notebook: `data/<trial>/`,
`notebooks/<trial>/`, and `tools/<trial>/` (when the trial has a data-prep script).

### `snips-intent-router` — recipe-generalization test

Checks whether the fine-tuning recipe itself generalizes (rather than being an artifact of
this one dataset) by running the same Qwen3-1.7B-Base + LoRA / LLaMA Board pipeline on the
[SNIPS NLU benchmark](https://github.com/sonos/nlu-benchmark) — a 7-intent voice-assistant
dataset with the same short-utterance, 7-label shape as the family router.

| File | What it is |
|---|---|
| `notebooks/snips-intent-router/finetune_snips_intent_router.ipynb` | Same recipe as the family-router notebook, applied to `data/snips-intent-router/snips_intent_routing.csv`. |
| `notebooks/snips-intent-router/finetune_snips_intent_router_executed.ipynb` | The same notebook as actually run, outputs included — evidence of the 96.7%-vs-79.1% result below. |
| `tools/snips-intent-router/prepare_snips_dataset.py` | Downloads the [`benayas/snips`](https://huggingface.co/datasets/benayas/snips) mirror and subsamples it to roughly match `family_request_routing.csv`'s per-label row count. |
| `data/snips-intent-router/snips_intent_routing.csv` / `_train.csv` / `_val.csv` / `_manifest.json` | Same file shapes as the family-request dataset. |

```
pip install -r requirements.txt
python tools/snips-intent-router/prepare_snips_dataset.py --per-class 65
```

**Result:** fine-tuned accuracy 96.7% vs. a 79.1% baseline (+17.6 pts) — the recipe
generalizes, though SNIPS's baseline is much higher than the family router's (~24–28%) since
its intents are mostly distinguishable by surface vocabulary alone. See the notebook's own
Recap section for the full breakdown.

## Relationship to family-calendar

This repo is intentionally standalone: the seed examples were pulled once from
[family-calendar](https://github.com/anushaakkiraju26/family-calendar)'s evaluation suite
(`evaluations/cases.json` + `evaluations/results_baseline.csv`) and committed here as
`data/seed_examples.json`, so this project doesn't depend on that repo at runtime.
