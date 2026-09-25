import numpy as np
import pytest

from espelho.cost import embedding_cost
from espelho.taxonomy import agreement_with_labels, cluster, detect_dish, purity


def test_detect_dish_with_accents_and_complaints():
    assert detect_dish("O RISOTO de camarão estava perfeito") == "risoto de camarão"
    assert detect_dish("Hambúrguer sem graça e mal preparado.") == "hambúrguer"
    assert detect_dish("A pizza estava ótima") is None


def test_purity_by_hand():
    clusters = np.array([0, 0, 0, 1, 1, -1])
    labels = ["a", "a", "b", "c", "c", "z"]
    # grupo 0: 2 de 3 são "a"; grupo 1: 2 de 2 são "c"; o ruído não conta
    assert purity(clusters, labels) == pytest.approx(4 / 5)


def test_cluster_finds_two_obvious_groups():
    rng = np.random.default_rng(0)
    group_a = rng.normal(0, 0.05, size=(30, 2))
    group_b = rng.normal(5, 0.05, size=(30, 2))
    labels = cluster(np.vstack([group_a, group_b]), min_cluster_size=10)
    result = agreement_with_labels(labels, ["a"] * 30 + ["b"] * 30)
    assert result["clusters"] == 2
    assert result["purity"] == pytest.approx(1.0)


def test_embedding_cost():
    assert embedding_cost("text-embedding-3-small", 1_000_000) == pytest.approx(0.02)
