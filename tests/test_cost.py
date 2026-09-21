import pytest

from espelho.cost import call_cost


def test_haiku_cost_input_and_output_are_charged_separately():
    # 1.000 tokens de entrada (US$ 1/M) + 1.000 de saída (US$ 5/M) = US$ 0,006
    assert call_cost("claude-haiku-4-5", 1_000, 1_000) == pytest.approx(0.006)


def test_batch_costs_half():
    normal = call_cost("claude-haiku-4-5", 1_000, 1_000)
    batch = call_cost("claude-haiku-4-5", 1_000, 1_000, batch=True)
    assert batch == pytest.approx(normal / 2)
