"""Aplica as correções do gabarito (corrections_v2.json) e gera dev_v2.parquet.

O dev.parquet original não é alterado: as duas versões ficam lado a lado.

Uso:
    uv run python scripts/apply_corrections.py
"""

import json
from pathlib import Path

import pandas as pd

CORRECTIONS = Path("data/gold/corrections_v2.json")
SOURCE = Path("data/gold/dev.parquet")
OUT = Path("data/gold/dev_v2.parquet")


def main() -> None:
    spec = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
    by_id = {
        c["review_id"]: [{"aspect": a, "polarity": p} for a, p in c["mentions"]]
        for c in spec["corrections"]
    }

    df = pd.read_parquet(SOURCE)
    unknown = set(by_id) - set(df["review_id"])
    if unknown:
        raise SystemExit(f"Ids que não estão no dev: {sorted(unknown)}")

    df["gold_mentions"] = [
        by_id.get(rid, list(mentions)) for rid, mentions in zip(df["review_id"], df["gold_mentions"])
    ]
    df["gold_n_mentions"] = df["gold_mentions"].map(len)
    df.to_parquet(OUT, index=False)
    print(f"{len(by_id)} reviews corrigidas -> {OUT}")


if __name__ == "__main__":
    main()
