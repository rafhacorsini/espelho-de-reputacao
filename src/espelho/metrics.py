"""Métricas de avaliação da extração, comparando previsões com o rótulo ouro.

Unidade de contagem: uma tupla (review, aspecto, polaridade). Uma previsão só
conta como acerto (verdadeiro positivo) se o aspecto E a polaridade batem.

- Verdadeiro positivo (TP): o sistema achou e estava no ouro.
- Falso positivo (FP): o sistema achou, mas não estava no ouro (inventou).
- Falso negativo (FN): estava no ouro, mas o sistema não achou (deixou passar).
"""

from collections import Counter

import numpy as np
from sklearn.metrics import cohen_kappa_score

from espelho.schema import ASPECTS


def as_tuples(mentions) -> set[tuple[str, str]]:
    return {(m["aspect"], m["polarity"]) for m in mentions}


def count_errors(gold_by_review: dict, pred_by_review: dict) -> dict:
    """TP, FP e FN por aspecto, somando todas as reviews."""
    counts = {aspect: Counter() for aspect in ASPECTS}
    for review_id, gold_mentions in gold_by_review.items():
        gold = as_tuples(gold_mentions)
        pred = as_tuples(pred_by_review.get(review_id, []))
        for aspect, polarity in gold & pred:
            counts[aspect]["tp"] += 1
        for aspect, polarity in pred - gold:
            counts[aspect]["fp"] += 1
        for aspect, polarity in gold - pred:
            counts[aspect]["fn"] += 1
    return counts


def prf(tp: int, fp: int, fn: int) -> dict:
    """Precisão, recall e F1. Sem dados, a métrica fica indefinida (None)."""
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None or precision + recall == 0:
        f1 = None if precision is None or recall is None else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


def evaluate(gold_by_review: dict, pred_by_review: dict) -> dict:
    """Métricas por aspecto, micro (soma tudo) e macro (média entre aspectos)."""
    counts = count_errors(gold_by_review, pred_by_review)
    per_aspect = {aspect: prf(c["tp"], c["fp"], c["fn"]) for aspect, c in counts.items()}

    total = sum(counts.values(), Counter())
    micro = prf(total["tp"], total["fp"], total["fn"])

    f1s = [m["f1"] for m in per_aspect.values() if m["f1"] is not None]
    macro_f1 = sum(f1s) / len(f1s) if f1s else None

    empty_gold = [rid for rid, mentions in gold_by_review.items() if not mentions]
    empty_kept = sum(1 for rid in empty_gold if not pred_by_review.get(rid))

    return {
        "per_aspect": per_aspect,
        "micro": micro,
        "macro_f1": macro_f1,
        "empty_reviews": len(empty_gold),
        "empty_correct": empty_kept,
    }


def bootstrap_f1(gold: dict, pred: dict, n_resamples: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    """F1 micro e o intervalo de 95%, sorteando as reviews com reposição."""
    ids = np.array(list(gold))
    rng = np.random.default_rng(seed)

    def f1(sample):
        return evaluate({i: gold[i] for i in sample}, {i: pred.get(i, []) for i in sample})["micro"]["f1"] or 0.0

    values = [f1(rng.choice(ids, size=len(ids), replace=True)) for _ in range(n_resamples)]
    low, high = np.percentile(values, [2.5, 97.5])
    return f1(ids), float(low), float(high)


def agreement(labels_a: dict, labels_b: dict) -> dict:
    """Concordância entre duas rotulagens das mesmas reviews.

    Cada par (review, aspecto) vira uma de três classes: nenhum, positivo ou
    negativo. O kappa de Cohen desconta a concordância que viria por acaso
    (como quase todo par é "nenhum", concordar nele é fácil e vale pouco).
    """
    shared = sorted(labels_a.keys() & labels_b.keys())
    a, b = [], []
    for review_id in shared:
        pol_a = {m["aspect"]: m["polarity"] for m in labels_a[review_id]}
        pol_b = {m["aspect"]: m["polarity"] for m in labels_b[review_id]}
        for aspect in ASPECTS:
            a.append(pol_a.get(aspect, "nenhum"))
            b.append(pol_b.get(aspect, "nenhum"))
    raw = sum(x == y for x, y in zip(a, b)) / len(a) if a else None
    kappa = cohen_kappa_score(a, b) if a else None
    return {"reviews": len(shared), "raw_agreement": raw, "kappa": kappa}


def polarity_confusion(gold_by_review: dict, pred_by_review: dict) -> dict:
    """Entre os aspectos que os dois acharam, quantas vezes a polaridade bateu."""
    matrix = Counter()
    for review_id, gold_mentions in gold_by_review.items():
        gold = {m["aspect"]: m["polarity"] for m in gold_mentions}
        pred = {m["aspect"]: m["polarity"] for m in pred_by_review.get(review_id, [])}
        for aspect in gold.keys() & pred.keys():
            matrix[(gold[aspect], pred[aspect])] += 1
    return dict(matrix)
