"""Impacto em estrelas e alarme de problemas novos (Dia 5).

Impacto: uma regressão linear responde "quando uma review menciona X, a nota
muda quanto, em média, mantendo o resto igual?". É associação, não causa.

Alarme (CUSUM): todo dia comparamos a taxa de um assunto com o normal da loja.
Pequenos excessos vão se somando; quando a soma passa de um limite, o alarme
toca, anota o dia e zera, para continuar vigiando. A soma nunca fica abaixo de zero.

Versão 2 do alarme (depois do resultado ruim da v1): para contagens raras, a conta
certa é a razão de verossimilhança binomial, não a aproximação normal, que fazia
uma única menção rara parecer um terremoto.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from espelho.schema import ASPECTS

CATEGORIES = [f"{a}|{p}" for a in ASPECTS for p in ("positivo", "negativo")]


def indicator_table(mentions: pd.Series) -> pd.DataFrame:
    """Uma linha por review, uma coluna 0/1 por aspecto|polaridade."""
    rows = [{f"{m['aspect']}|{m['polarity']}": 1 for m in ms} for ms in mentions]
    return pd.DataFrame(rows, index=mentions.index).reindex(columns=CATEGORIES).fillna(0).astype(int)


def fit_impact(indicators: pd.DataFrame, stars: pd.Series) -> pd.DataFrame:
    """Coeficiente de cada categoria (estrelas a mais ou a menos) com intervalo de 95%."""
    X = sm.add_constant(indicators.astype(float))
    model = sm.OLS(stars.astype(float), X).fit()
    ci = model.conf_int(0.05)
    table = pd.DataFrame({"coef": model.params, "low": ci[0], "high": ci[1]}).drop(index="const")
    table["intercept"] = model.params["const"]
    return table


def binomial_cusum(
    counts: np.ndarray, totals: np.ndarray, baseline_days: int, h: float = 5.0, min_shift: float = 0.10
) -> list[int]:
    """Dias em que o alarme de AUMENTO de uma taxa disparou (zera depois de cada alarme).

    p0 é o normal, medido nos primeiros dias (com +1/+2 para não dar zero).
    p1 é a mudança que vale um alarme: dobrar, ou subir 10 pontos, o que for menor.
    Cada dia soma o quanto os dados combinam mais com p1 do que com p0.
    """
    p0 = (counts[:baseline_days].sum() + 1) / (totals[:baseline_days].sum() + 2)
    p1 = min(p0 + min(p0, min_shift), 0.99)
    up, down = np.log(p1 / p0), np.log((1 - p1) / (1 - p0))
    s, alarms = 0.0, []
    for day in range(baseline_days, len(counts)):
        s = max(0.0, s + counts[day] * up + (totals[day] - counts[day]) * down)
        if s > h:
            alarms.append(day)
            s = 0.0
    return alarms


def mean_cusum_down(values: list[np.ndarray], baseline_days: int, k: float = 0.5, h: float = 5.0) -> list[int]:
    """Dias em que o alarme de QUEDA de uma média diária (a nota média) disparou."""
    base = np.concatenate([v for v in values[:baseline_days] if len(v)])
    mu, sd = base.mean(), base.std(ddof=1)
    s, alarms = 0.0, []
    for day in range(baseline_days, len(values)):
        v = values[day]
        if len(v):
            s = max(0.0, s + (mu - v.mean()) / (sd / np.sqrt(len(v))) - k)
        if s > h:
            alarms.append(day)
            s = 0.0
    return alarms
