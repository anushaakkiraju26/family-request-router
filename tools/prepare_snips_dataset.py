"""Prepare a SNIPS-based dataset to test the same fine-tuning recipe on a
second, unrelated task.

Task: given one short natural-language utterance, predict which of seven
voice-assistant intents it belongs to. This is the classic SNIPS NLU
benchmark intent set (AddToPlaylist, BookRestaurant, GetWeather, PlayMusic,
RateBook, SearchCreativeWork, SearchScreeningEvent) — chosen specifically
because it has the same *shape* as this project's family-request router
(short request -> one of seven labels), so it's a reasonably direct test of
whether the Qwen3-1.7B-Base + LoRA / LLaMA Board recipe generalizes to a
different label set, rather than a test of a differently-shaped problem.

Source: the `benayas/snips` mirror on the Hugging Face Hub of the original
SNIPS NLU benchmark (sonos/nlu-benchmark), already deduplicated into a flat
`text`/`category` CSV shape (13,084 train rows, 1,400 test rows, both
balanced across the 7 intents).

This script downloads that data (via the HF `parquet` API, no `datasets`
library dependency), subsamples it down to a size comparable to
data/family_request_routing.csv (so the two recipes are trained on similar
data volume, not just similar label-set size), and writes it out in the
same file shapes as the family-request dataset:

  data/snips_intent_routing.csv       (text, category_truth, source) — all rows
  data/snips_intent_routing_train.csv (80%, stratified)
  data/snips_intent_routing_val.csv   (20%, stratified)
  data/snips_intent_routing_manifest.json (counts + provenance)

Needs `pyarrow` and `requests` (see requirements.txt).

Run:

  python tools/prepare_snips_dataset.py --per-class 65
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

HF_DATASET = "benayas/snips"
HF_PARQUET_API = f"https://huggingface.co/api/datasets/{HF_DATASET}/parquet"

LABELS = [
    "AddToPlaylist",
    "BookRestaurant",
    "GetWeather",
    "PlayMusic",
    "RateBook",
    "SearchCreativeWork",
    "SearchScreeningEvent",
]


def _fetch_split(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_parquet(io.BytesIO(resp.content))


def fetch_snips() -> pd.DataFrame:
    """Download and concatenate the benayas/snips train+test splits."""
    manifest = requests.get(HF_PARQUET_API, timeout=30).json()
    urls = manifest["default"]["train"] + manifest["default"]["test"]
    frames = [_fetch_split(u) for u in urls]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"category": "category_truth"})
    df = df[df["category_truth"].isin(LABELS)].reset_index(drop=True)
    df["source"] = "snips_benchmark"
    return df[["text", "category_truth", "source"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-class",
        type=int,
        default=65,
        help="Rows to keep per label after subsampling (default: 65, "
        "matching the family-request dataset's rough per-label density).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Fetching {HF_DATASET} from the Hugging Face Hub...")
    full_df = fetch_snips()
    print(f"Fetched {len(full_df):,} total rows across {full_df['category_truth'].nunique()} labels")

    # Stratified subsample: same --per-class rows for every label, so the
    # class balance matches the family-request dataset's near-uniform split
    # rather than inheriting SNIPS's own (already fairly balanced) counts.
    subsamples = [
        group.sample(n=min(args.per_class, len(group)), random_state=args.seed)
        for _, group in full_df.groupby("category_truth")
    ]
    df = (
        pd.concat(subsamples, ignore_index=True)
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)
    )
    print(f"\nSubsampled to {len(df):,} rows:")
    print(df["category_truth"].value_counts())

    df_train, df_val = train_test_split(
        df, test_size=0.2, stratify=df["category_truth"], random_state=args.seed,
    )
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    DATA_DIR.mkdir(exist_ok=True)
    full_path = DATA_DIR / "snips_intent_routing.csv"
    train_path = DATA_DIR / "snips_intent_routing_train.csv"
    val_path = DATA_DIR / "snips_intent_routing_val.csv"
    manifest_path = DATA_DIR / "snips_intent_routing_manifest.json"

    df.to_csv(full_path, index=False)
    df_train[["text", "category_truth"]].to_csv(train_path, index=False)
    df_val[["text", "category_truth"]].to_csv(val_path, index=False)

    manifest = {
        "purpose": (
            "Second dataset to test the family-request-router's fine-tuning "
            "recipe (Qwen3-1.7B-Base + LoRA via LLaMA-Factory's LLaMA Board) "
            "on an unrelated, well-known intent-classification task with the "
            "same 7-label shape, as a generalization sanity check."
        ),
        "labels": LABELS,
        "counts_total": df["category_truth"].value_counts().to_dict(),
        "train_count": len(df_train),
        "val_count": len(df_val),
        "per_class_requested": args.per_class,
        "source": {
            "hf_dataset": HF_DATASET,
            "original_benchmark": "SNIPS NLU benchmark (sonos/nlu-benchmark)",
            "hf_total_rows_available": len(full_df),
            "note": (
                "benayas/snips is a flat text/category mirror of the original "
                "SNIPS NLU benchmark's per-intent JSON files. Subsampled here "
                "to roughly match data/family_request_routing.csv's per-label "
                "row count so the two recipes train on comparable data volume."
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote:\n  {full_path}\n  {train_path}\n  {val_path}\n  {manifest_path}")


if __name__ == "__main__":
    main()
