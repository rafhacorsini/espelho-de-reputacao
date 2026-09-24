"""Intervalo de confiança do F1 por bootstrap: sorteia as reviews do test com
reposição milhares de vezes e vê quanto o F1 varia.

Com só 50 reviews, uma ou duas menções mudam o F1. O intervalo mostra se a
diferença entre dois sistemas é real ou está dentro do ruído.

Uso:
    uv run python scripts/bootstrap_ci.py
"""

import numpy as np
import pandas as pd

from espelho.baseline import extract_baseline
from espelho.metrics import evaluate

N_RESAMPLES = 2000
SEED = 7


def micro_f1(gold: dict, pred: dict, ids) -> float:
    return evaluate({i: gold[i] for i in ids}, {i: pred[i] for i in ids})["micro"]["f1"] or 0.0


def main() -> None:
    gold_df = pd.read_parquet("data/gold/test_v2.parquet")
    gold = {r.review_id: list(r.gold_mentions) for r in gold_df.itertuples()}
    systems = {
        "extract_v2": {r.review_id: list(r.pred_mentions) for r in pd.read_parquet("data/predictions/test_extract_v2.parquet").itertuples()},
        "extract_v1": {r.review_id: list(r.pred_mentions) for r in pd.read_parquet("data/predictions/test_extract_v1.parquet").itertuples()},
        "baseline": {r.review_id: extract_baseline(r.text) for r in gold_df.itertuples()},
    }

    ids = np.array(list(gold))
    rng = np.random.default_rng(SEED)
    samples = {name: [] for name in systems}
    diffs = []
    for _ in range(N_RESAMPLES):
        sample = rng.choice(ids, size=len(ids), replace=True)
        scores = {name: micro_f1(gold, pred, sample) for name, pred in systems.items()}
        for name, score in scores.items():
            samples[name].append(score)
        diffs.append(scores["extract_v2"] - scores["extract_v1"])

    print(f"F1 micro no test (50 reviews), intervalo de 95% com {N_RESAMPLES} reamostragens:")
    for name, values in samples.items():
        point = micro_f1(gold, systems[name], ids)
        low, high = np.percentile(values, [2.5, 97.5])
        print(f"  {name:<11} {point:.2f}  [{low:.2f} a {high:.2f}]")
    low, high = np.percentile(diffs, [2.5, 97.5])
    print(f"\nDiferença v2 - v1: [{low:+.2f} a {high:+.2f}]  (se o intervalo cruza o zero, não dá para dizer que um é melhor)")


if __name__ == "__main__":
    main()
