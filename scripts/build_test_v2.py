"""Monta o gabarito final do test: rodada cega + conciliação (test_v2_adjudication.json).

Uso:
    uv run python scripts/build_test_v2.py
"""

import json
from pathlib import Path

import pandas as pd

from espelho.gold import load_labels

BLIND = Path("data/gold/relabel_test.jsonl")
ADJUDICATION = Path("data/gold/test_v2_adjudication.json")
SOURCE = Path("data/gold/test.parquet")
OUT = Path("data/gold/test_v2.parquet")


def main() -> None:
    blind = load_labels(BLIND)
    spec = json.loads(ADJUDICATION.read_text(encoding="utf-8"))

    mentions = {rid: list(record["mentions"]) for rid, record in blind.items()}
    for adj in spec["adjustments_to_blind_labels"]:
        aspect, polarity = adj["add"]
        current = mentions[adj["review_id"]]
        if not any(m["aspect"] == aspect for m in current):
            current.append({"aspect": aspect, "polarity": polarity})

    df = pd.read_parquet(SOURCE)
    missing = set(df["review_id"]) - set(mentions)
    if missing:
        raise SystemExit(f"Faltam na rodada cega: {sorted(missing)}")

    df["gold_mentions"] = df["review_id"].map(mentions)
    df["gold_n_mentions"] = df["gold_mentions"].map(len)
    df.to_parquet(OUT, index=False)
    print(f"{len(df)} reviews -> {OUT}")


if __name__ == "__main__":
    main()
