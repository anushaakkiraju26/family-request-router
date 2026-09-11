# Family Request Router — Week 5 Fine-Tuning Project

Custom-dataset variant of The Gen Academy's [Fine-Tune a Support Ticket Router](https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router)
Week 5 project. Same recipe — `Qwen/Qwen3-1.7B-Base` + LoRA via LLaMA-Factory's LLaMA Board
UI — applied to my own data instead of IT support tickets.

## What it does

Classifies a parent's request to a family-calendar coordinator agent (e.g. *"Add Leo's soccer
practice tomorrow from 4 to 5 PM for family-1"*) into one of six routing labels the
coordinator in my [family-calendar](https://github.com/anushaakkiraju26/family-calendar)
project uses to decide how to handle a request:

| Label | Downstream action |
|---|---|
| `fast_path_mutate` | Direct create/update/move/delete/restore, pending approval |
| `fast_path_read` | Direct list/show, no mutation |
| `fast_path_reject` | Must be refused by deterministic safeguards (past event, conflict, cross-family, stale version, unsafe retry) |
| `deep_weekly_workflow` | Full specialist pipeline: intake → planner → transportation → reviewer → reminder |
| `outing_workflow` | Constrained outing/activity search wrapper, never mutates the calendar |
| `ambiguous_clarify` | Missing info — must ask before acting |

## Files

| File | What it is |
|---|---|
| `notebooks/finetune_family_request_router.ipynb` | The Colab notebook — install, prepare data, train via LLaMA Board, merge, evaluate. Open it directly in Colab. |
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

## Relationship to family-calendar

This repo is intentionally standalone: the seed examples were pulled once from
[family-calendar](https://github.com/anushaakkiraju26/family-calendar)'s evaluation suite
(`evaluations/cases.json` + `evaluations/results_baseline.csv`) and committed here as
`data/seed_examples.json`, so this project doesn't depend on that repo at runtime.

## Submission

Per the Week 5 handout: using a custom dataset means submitting a GitHub link with all assets
plus a short Loom video, instead of the Google Doc + screenshot route.
