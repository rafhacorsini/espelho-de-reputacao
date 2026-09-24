"""Trava de regressão da qualidade (Dia 3, passo 9).

Usa as previsões salvas do dev, sem chamar a API. Se alguém gerar previsões
novas com um prompt pior, este teste falha. Usa o dev, nunca o test: o test é
só para a nota final.
"""

from pathlib import Path

import pandas as pd
import pytest

from espelho.metrics import evaluate

GOLD = Path("data/gold/dev_v2.parquet")
PREDICTIONS = Path("data/predictions/dev_extract_v2.parquet")
MIN_MICRO_F1 = 0.90
MIN_EMPTY_CORRECT = 0.85

pytestmark = pytest.mark.skipif(
    not (GOLD.exists() and PREDICTIONS.exists()), reason="faltam o gabarito ou as previsões do dev"
)


def test_dev_quality_does_not_regress():
    gold_df = pd.read_parquet(GOLD)
    pred_df = pd.read_parquet(PREDICTIONS)
    gold = {r.review_id: list(r.gold_mentions) for r in gold_df.itertuples()}
    pred = {r.review_id: list(r.pred_mentions) for r in pred_df.itertuples()}

    result = evaluate(gold, pred)
    assert result["micro"]["f1"] >= MIN_MICRO_F1
    assert result["empty_correct"] / result["empty_reviews"] >= MIN_EMPTY_CORRECT


def test_every_evidence_is_verbatim():
    pred_df = pd.read_parquet(PREDICTIONS)
    mentions = [m for row in pred_df["pred_mentions"] for m in row]
    assert all(m["evidence_verified"] for m in mentions)
