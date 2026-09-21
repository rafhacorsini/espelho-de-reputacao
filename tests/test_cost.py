import pytest

from espelho.cost import call_cost


def test_luna_cost_input_and_output_are_charged_separately():
    # 1.000 tokens de entrada (US$ 0,20/M) + 1.000 de saída (US$ 1,20/M) = US$ 0,0014
    assert call_cost("gpt-5.6-luna", 1_000, 1_000) == pytest.approx(0.0014)


def test_batch_costs_half():
    normal = call_cost("gpt-5.6-luna", 1_000, 1_000)
    batch = call_cost("gpt-5.6-luna", 1_000, 1_000, batch=True)
    assert batch == pytest.approx(normal / 2)
