"""Dia 5: quanto cada dor custa em estrelas, e o alarme de problemas novos.

Alarme v2 (a v1, com aproximação normal, deu 38 alarmes falsos em 112 séries;
resultado guardado em reports/day5_results_alarm_v1.json). Como a v2 foi
desenhada depois de ver a v1 neste corpus, ela é validada num mundo novo:
reviews sorteadas com outra semente (VALIDATION_SEED), que o alarme nunca viu.

Parâmetros: 15 dias de "normal", limite h = 5, mudança que vale alarme =
dobrar ou subir 10 pontos. Um evento conta como achado se o alarme tocar na
série dele em até 14 dias depois do início.

Uso:
    uv run python scripts/impact_and_alarms.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from espelho.analytics import CATEGORIES, binomial_cusum, fit_impact, indicator_table, mean_cusum_down
from espelho.synth.truth import EVENTS, NEGATIVE_EFFECT, POSITIVE_EFFECT, generate_truth

CORPUS = Path("data/synthetic/corpus_reviews.parquet")
PREDICTIONS = Path("data/predictions/corpus_extract_v2.parquet")
SNIPPETS = Path("data/taxonomy/snippets_v1.parquet")
REPORT = Path("reports/owner_report.md")
RESULTS = Path("reports/day5_results.json")

BASELINE_DAYS, H = 15, 5.0
H_SENSITIVITY = [3.0, 5.0, 8.0]
DETECTION_WINDOW = 14
VALIDATION_SEED = 2027
RECENT_DAYS = 14
NETWORK = "Rede"
NEGATIVE_EVENTS = [e for e in EVENTS if e.polarity == "negativo"]


def planted_effect(category: str) -> float:
    aspect, polarity = category.split("|")
    return (NEGATIVE_EFFECT if polarity == "negativo" else POSITIVE_EFFECT)[aspect]


def impact(corpus: pd.DataFrame) -> pd.DataFrame:
    truth_ind = indicator_table(corpus["mentions"].map(json.loads))
    pred_ind = indicator_table(corpus["pred_mentions"])
    method_check = fit_impact(truth_ind, corpus["stars_latent"])
    pipeline = fit_impact(pred_ind, corpus["stars"])
    table = pd.DataFrame(
        {
            "plantado": [planted_effect(c) for c in CATEGORIES],
            "verdade_latente": method_check["coef"].reindex(CATEGORIES).values,
            "pipeline": pipeline["coef"].reindex(CATEGORIES).values,
            "pipeline_low": pipeline["low"].reindex(CATEGORIES).values,
            "pipeline_high": pipeline["high"].reindex(CATEGORIES).values,
        },
        index=CATEGORIES,
    )
    return table


def daily_series(corpus: pd.DataFrame, indicators: pd.DataFrame, unit: str, category: str):
    rows = corpus if unit == NETWORK else corpus[corpus["store_name"] == unit]
    days = np.arange(corpus["day"].max() + 1)
    totals = rows.groupby("day").size().reindex(days, fill_value=0).to_numpy()
    counts = indicators.loc[rows.index, category].groupby(rows["day"]).sum().reindex(days, fill_value=0).to_numpy()
    return counts, totals


def event_unit(event) -> str:
    return NETWORK if event.store_id is None else f"Loja {event.store_id}"


def affecting_events(unit: str, category: str):
    """Eventos que mexem nesta série (loja × categoria)."""
    found = []
    for e in EVENTS:
        if f"{e.aspect}|{e.polarity}" != category:
            continue
        if unit == NETWORK or e.store_id is None or f"Loja {e.store_id}" == unit:
            found.append(e)
    return found


def first_in_window(alarms: list[int], start: int) -> int | None:
    return next((a for a in alarms if start <= a <= start + DETECTION_WINDOW), None)


def run_alarms(corpus: pd.DataFrame, indicators: pd.DataFrame, h: float) -> dict:
    """Vigia todas as séries (loja × categoria, mais a rede toda) e mede acertos e alarmes falsos."""
    units = sorted(corpus["store_name"].unique()) + [NETWORK]
    detections, false_alarms, series = {}, 0, 0
    for unit in units:
        for category in CATEGORIES:
            counts, totals = daily_series(corpus, indicators, unit, category)
            alarms = binomial_cusum(counts, totals, BASELINE_DAYS, h)
            events = affecting_events(unit, category)
            series += 1
            first_start = min((e.start_day for e in events), default=None)
            false_alarms += sum(1 for a in alarms if first_start is None or a < first_start)
            for e in events:
                if unit == event_unit(e):
                    hit = first_in_window(alarms, e.start_day)
                    if hit is not None:
                        detections[e.event_id] = hit
    series_days = series * (corpus["day"].max() + 1 - BASELINE_DAYS)
    return {
        "detections": detections,
        "false_alarms": false_alarms,
        "series": series,
        "false_per_1000_series_days": 1000 * false_alarms / series_days,
    }


def star_alarms(corpus: pd.DataFrame) -> dict:
    """O jeito comum: vigiar só a queda da nota média, por loja e na rede."""
    units = sorted(corpus["store_name"].unique()) + [NETWORK]
    detections, false_alarms = {}, 0
    for unit in units:
        rows = corpus if unit == NETWORK else corpus[corpus["store_name"] == unit]
        values = [rows.loc[rows["day"] == d, "stars"].to_numpy(dtype=float) for d in range(corpus["day"].max() + 1)]
        alarms = mean_cusum_down(values, BASELINE_DAYS, h=H)
        starts = [e.start_day for e in NEGATIVE_EVENTS if unit == NETWORK or e.store_id is None or event_unit(e) == unit]
        false_alarms += sum(1 for a in alarms if a < min(starts, default=10**9))
        for e in EVENTS:
            if event_unit(e) == unit:
                hit = first_in_window(alarms, e.start_day)
                if hit is not None:
                    detections[e.event_id] = hit
    return {"detections": detections, "false_alarms": false_alarms}


def summarize(run: dict) -> str:
    delays = [run["detections"][e.event_id] - e.start_day for e in EVENTS if e.event_id in run["detections"]]
    mean_delay = f"{np.mean(delays):.1f} dias" if delays else "-"
    return (
        f"achou {len(delays)}/5 | atraso médio {mean_delay} | alarmes falsos {run['false_alarms']} "
        f"({run['false_per_1000_series_days']:.1f} por 1.000 séries-dia)"
    )


def owner_report(corpus: pd.DataFrame, indicators: pd.DataFrame, impact_table: pd.DataFrame) -> str:
    snippets = pd.read_parquet(SNIPPETS).merge(corpus[["review_id", "store_name", "day"]], on="review_id")
    recent_from = corpus["day"].max() - RECENT_DAYS + 1
    lines = [
        "# Relatório para o dono (Dia 5)",
        "",
        f"Últimos {RECENT_DAYS} dias (dia {recent_from} ao {corpus['day'].max()}). "
        "\"Estrelas em jogo por 100 reviews\" = impacto de cada menção × quantas reviews a mencionam. "
        "É associação, não causa.",
    ]
    for store in sorted(corpus["store_name"].unique()):
        rows = corpus[(corpus["store_name"] == store) & (corpus["day"] >= recent_from)]
        share = indicators.loc[rows.index].mean()
        weight = (impact_table["pipeline"] * share * 100).dropna()
        pains = weight[weight < 0].sort_values().head(3)
        strengths = weight[weight > 0].sort_values(ascending=False).head(3)
        store_snips = snippets[(snippets["store_name"] == store) & (snippets["day"] >= recent_from)]

        lines += ["", f"## {store} ({len(rows)} reviews, nota média {rows['stars'].mean():.2f})", "", "**Dores que mais custam:**"]
        for category, value in pains.items():
            example = store_snips[store_snips["category"] == category]["text"].head(1).tolist()
            quote = f" — \"{example[0]}\"" if example else ""
            lines.append(f"- {category.replace('|', ', ')}: {value:.1f} estrelas por 100 reviews ({share[category]:.0%} das reviews){quote}")
        lines += ["", "**Forças para o marketing:**"]
        for category, value in strengths.items():
            example = store_snips[store_snips["category"] == category]["text"].head(1).tolist()
            quote = f" — \"{example[0]}\"" if example else ""
            lines.append(f"- {category.replace('|', ', ')}: +{value:.1f} estrelas por 100 reviews{quote}")
        dishes = store_snips["dish"].dropna().value_counts()
        for dish, n in dishes.items():
            praise = ((store_snips["dish"] == dish) & (store_snips["polarity"] == "positivo")).sum()
            lines.append(f"- Prato em destaque: {dish} ({praise} elogios e {n - praise} reclamações no período)")
    return "\n".join(lines) + "\n"


def main() -> None:
    corpus = pd.read_parquet(CORPUS).merge(pd.read_parquet(PREDICTIONS)[["review_id", "pred_mentions"]], on="review_id")
    corpus = corpus.reset_index(drop=True)
    indicators = indicator_table(corpus["pred_mentions"])

    table = impact(corpus)
    print("=== Quanto cada menção muda a nota (estrelas) ===")
    print(f"{'categoria':<30} {'plantado':>8} {'método*':>8} {'pipeline':>9}  intervalo 95%")
    for category, r in table.iterrows():
        print(
            f"{category:<30} {r['plantado']:>8.2f} {r['verdade_latente']:>8.2f} {r['pipeline']:>9.2f}"
            f"  [{r['pipeline_low']:.2f} a {r['pipeline_high']:.2f}]"
        )
    print("* método: menções verdadeiras e nota sem arredondar (confere se a régua funciona)")

    stars = star_alarms(corpus)
    main_run = run_alarms(corpus, indicators, H)
    print(f"\n=== Alarme v2 no nosso corpus (h={H}, {BASELINE_DAYS} dias de normal) ===")
    print(f"{'evento':<44} {'início':>6} {'aspectos':>9} {'nota média':>11}")
    for e in EVENTS:
        alarm, star = main_run["detections"].get(e.event_id), stars["detections"].get(e.event_id)
        a_txt = f"+{alarm - e.start_day} dias" if alarm is not None else "não avisou"
        s_txt = f"+{star - e.start_day} dias" if star is not None else "não avisou"
        print(f"{e.event_id} {e.description:<40} {e.start_day:>6} {a_txt:>9} {s_txt:>11}")
    print(f"Aspectos:   {summarize(main_run)} em {main_run['series']} séries")
    print(f"Nota média: achou {len(stars['detections'])}/5 | alarmes falsos {stars['false_alarms']} em 7 séries")

    validation = generate_truth(seed=VALIDATION_SEED)
    validation_ind = indicator_table(validation["mentions"].map(json.loads))
    validation_run = run_alarms(validation, validation_ind, H)
    print(f"\n=== Validação num mundo novo (semente {VALIDATION_SEED}, que o alarme nunca viu) ===")
    print(f"Aspectos:   {summarize(validation_run)}")
    print("(menções verdadeiras, como se o extrator fosse perfeito: mede o alarme, não o extrator)")

    print("\n=== A troca: limite h mais baixo avisa antes, mas erra mais (nosso corpus) ===")
    sensitivity = {}
    for h in H_SENSITIVITY:
        run = run_alarms(corpus, indicators, h)
        sensitivity[h] = {k: run[k] for k in ("detections", "false_alarms", "false_per_1000_series_days")}
        print(f"  h={h:<4} {summarize(run)}")

    REPORT.write_text(owner_report(corpus, indicators, table), encoding="utf-8")
    RESULTS.write_text(
        json.dumps(
            {
                "impact": table.round(3).reset_index(names="category").to_dict(orient="records"),
                "alarms": main_run,
                "star_alarms": stars,
                "validation": validation_run,
                "sensitivity": sensitivity,
            },
            ensure_ascii=False, indent=2, default=int,
        ),
        encoding="utf-8",
    )
    print(f"\nRelatório para o dono -> {REPORT}")


if __name__ == "__main__":
    main()
