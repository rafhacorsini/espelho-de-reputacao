import pytest

from espelho.metrics import agreement, bootstrap_f1, evaluate, polarity_confusion, prf


def test_bootstrap_f1_perfect_prediction_has_tight_interval():
    gold = {f"R{i}": [{"aspect": "atendimento", "polarity": "positivo"}] for i in range(20)}
    point, low, high = bootstrap_f1(gold, gold, n_resamples=200)
    assert point == low == high == pytest.approx(1.0)


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


def test_agreement_identical_labels_is_perfect():
    labels = {"R1": [m("atendimento", "positivo")], "R2": [m("preco_valor", "negativo")]}
    result = agreement(labels, labels)
    assert result["raw_agreement"] == 1.0
    assert result["kappa"] == pytest.approx(1.0)


def test_agreement_kappa_is_below_raw_when_labels_differ():
    a = {"R1": [m("atendimento", "positivo")], "R2": [m("preco_valor", "negativo")]}
    b = {"R1": [m("atendimento", "positivo")], "R2": []}
    result = agreement(a, b)
    assert result["raw_agreement"] == pytest.approx(15 / 16)
    assert result["kappa"] < result["raw_agreement"]


def test_polarity_confusion():
    gold = {"R1": [m("atendimento", "positivo"), m("preco_valor", "negativo")]}
    pred = {"R1": [m("atendimento", "positivo"), m("preco_valor", "positivo")]}
    assert polarity_confusion(gold, pred) == {
        ("positivo", "positivo"): 1,
        ("negativo", "positivo"): 1,
    }
