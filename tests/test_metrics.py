import pytest

from espelho.metrics import evaluate, polarity_confusion, prf


def m(aspect, polarity):
    return {"aspect": aspect, "polarity": polarity}


def test_prf_by_hand():
    # 8 acertos, 2 inventados, 4 perdidos
    result = prf(tp=8, fp=2, fn=4)
    assert result["precision"] == pytest.approx(0.8)
    assert result["recall"] == pytest.approx(8 / 12)
    assert result["f1"] == pytest.approx(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12))


def test_prf_undefined_without_data():
    assert prf(0, 0, 0)["f1"] is None


def test_wrong_polarity_counts_as_one_fp_and_one_fn():
    gold = {"R1": [m("preco_valor", "negativo")]}
    pred = {"R1": [m("preco_valor", "positivo")]}
    result = evaluate(gold, pred)["per_aspect"]["preco_valor"]
    assert result["precision"] == 0 and result["recall"] == 0


def test_perfect_prediction():
    gold = {"R1": [m("atendimento", "positivo")], "R2": []}
    result = evaluate(gold, gold)
    assert result["micro"]["f1"] == pytest.approx(1.0)
    assert result["empty_correct"] == 1


def test_invented_aspect_on_empty_review():
    gold = {"R1": []}
    pred = {"R1": [m("condicao_entrega", "positivo")]}
    result = evaluate(gold, pred)
    assert result["empty_reviews"] == 1
    assert result["empty_correct"] == 0
    assert result["micro"]["precision"] == 0


def test_polarity_confusion():
    gold = {"R1": [m("atendimento", "positivo"), m("preco_valor", "negativo")]}
    pred = {"R1": [m("atendimento", "positivo"), m("preco_valor", "positivo")]}
    assert polarity_confusion(gold, pred) == {
        ("positivo", "positivo"): 1,
        ("negativo", "positivo"): 1,
    }
