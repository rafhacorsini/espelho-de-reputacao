import numpy as np
import pandas as pd

from espelho.analytics import binomial_cusum, fit_impact, indicator_table, mean_cusum_down


def test_indicator_table():
    mentions = pd.Series([[{"aspect": "preco_valor", "polarity": "negativo"}], []])
    table = indicator_table(mentions)
    assert table.loc[0, "preco_valor|negativo"] == 1
    assert table.loc[1].sum() == 0


def test_fit_impact_recovers_known_effect():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=2000)
    stars = 4.0 - 1.2 * x + rng.normal(0, 0.3, size=2000)
    table = fit_impact(pd.DataFrame({"prazo_entrega|negativo": x}), pd.Series(stars))
    row = table.loc["prazo_entrega|negativo"]
    assert row["low"] < -1.2 < row["high"]


def test_binomial_cusum_catches_a_jump():
    rng = np.random.default_rng(1)
    totals = np.full(60, 10)
    jump = np.concatenate([rng.binomial(10, 0.06, size=30), rng.binomial(10, 0.30, size=30)])
    alarms = binomial_cusum(jump, totals, baseline_days=15)
    assert alarms and 30 <= alarms[0] <= 36


def test_binomial_cusum_rarely_fires_on_flat_rare_series():
    # 200 séries sem mudança nenhuma, com um assunto raro (3%): quase nenhum alarme
    rng = np.random.default_rng(2)
    totals = np.full(60, 10)
    fired = sum(bool(binomial_cusum(rng.binomial(10, 0.03, size=60), totals, 15)) for _ in range(200))
    assert fired <= 10


def test_alarm_resets_and_keeps_watching():
    counts = np.array([0] * 15 + [5] * 10 + [0] * 10 + [5] * 10)
    alarms = binomial_cusum(counts, np.full(45, 10), baseline_days=15)
    assert any(a < 25 for a in alarms) and any(a >= 35 for a in alarms)


def test_mean_cusum_down_catches_a_drop():
    rng = np.random.default_rng(3)
    days = [rng.normal(4.0, 0.8, size=10) for _ in range(30)] + [rng.normal(3.2, 0.8, size=10) for _ in range(30)]
    alarms = mean_cusum_down(days, baseline_days=15)
    assert alarms and alarms[0] >= 30
