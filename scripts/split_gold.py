"""Junta os rótulos com o texto e divide em dev (para ajustar) e test (só no fim).

Uso:
    uv run python scripts/split_gold.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from espelho.gold import load_labels

PILOT_REVIEWS = Path("data/synthetic/pilot_reviews.parquet")
LABELS_PATH = Path("data/gold/labels.jsonl")
DEV_OUT = Path("data/gold/dev.parquet")
TEST_OUT = Path("data/gold/test.parquet")

TEST_SIZE = 50
SEED = 2026


def main() -> None:
    reviews = pd.read_parquet(PILOT_REVIEWS)
    labels = load_labels(LABELS_PATH)

    kept_ids = [rid for rid, record in labels.items() if not record.get("discarded")]
    gold = reviews[reviews["review_id"].isin(kept_ids)].copy()
    gold["gold_mentions"] = gold["review_id"].map(lambda rid: labels[rid]["mentions"])
    gold = gold.rename(columns={"mentions": "sorted_truth_mentions"})
    gold["gold_n_mentions"] = gold["gold_mentions"].map(len)

    # Estratifica por 0 / 1 / 2+ menções, para dev e test terem dificuldade parecida.
    strata = gold["gold_n_mentions"].clip(upper=2)
    dev, test = train_test_split(
        gold, test_size=TEST_SIZE, random_state=SEED, stratify=strata
    )

    columns = [
        "review_id", "store_id", "store_name", "channel", "stars", "text",
        "sorted_truth_mentions", "gold_mentions", "gold_n_mentions",
    ]
    dev[columns].reset_index(drop=True).to_parquet(DEV_OUT, index=False)
    test[columns].reset_index(drop=True).to_parquet(TEST_OUT, index=False)

    print(f"dev: {len(dev)} reviews -> {DEV_OUT}")
    print(f"test: {len(test)} reviews -> {TEST_OUT}")
    print("\ndistribuição de menções (0 / 1 / 2+):")
    for name, split in [("dev", dev), ("test", test)]:
        counts = split["gold_n_mentions"].clip(upper=2).value_counts().sort_index()
        print(f"  {name}: {counts.to_dict()}")


if __name__ == "__main__":
    main()
