"""Exporta os números do case para o site (site/src/data/case.json).

O site é estático: não roda Python nem chama API. Tudo que ele mostra sai daqui.

Uso:
    uv run python scripts/export_site_data.py
"""

import json
from pathlib import Path

import pandas as pd

from espelho.analytics import event_series, indicator_table
from espelho.synth.truth import EVENTS

OUT = Path("site/src/data/case.json")
SUMMARY = Path("reports/dashboard_summary.json")


def dish_story(corpus: pd.DataFrame) -> dict:
    snippets = pd.read_parquet("data/taxonomy/snippets_v1.parquet").merge(
        corpus[["review_id", "store_name", "day"]], on="review_id")
    risotto = snippets[snippets["cluster"] == 10]
    burger = snippets[snippets["dish"] == "hambúrguer"]
    return {
        "risotto_snippets": int(len(risotto)),
        "risotto_share_store2": float((risotto["store_name"] == "Loja 2").mean()),
        "risotto_share_after_day25": float((risotto["day"] >= 25).mean()),
        "burger_praise": int((burger["polarity"] == "positivo").sum()),
        "burger_complaints": int((burger["polarity"] == "negativo").sum()),
        "snippets_total": int(len(snippets)),
    }


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    corpus = pd.read_parquet("data/synthetic/corpus_reviews.parquet").merge(
        pd.read_parquet("data/predictions/corpus_extract_v2.parquet")[["review_id", "pred_mentions"]], on="review_id"
    ).reset_index(drop=True)
    indicators = indicator_table(corpus["pred_mentions"])

    events = []
    for e in EVENTS:
        s = event_series(corpus, indicators, e)
        events.append({
            "id": e.event_id, "description": e.description, "store": "Rede toda" if e.store_id is None else f"Loja {e.store_id}",
            "aspect": e.aspect, "polarity": e.polarity, "start": s["start"],
            "aspect_alarm": s["aspect_alarm"], "star_alarm": s["star_alarm"],
            "series": [{"day": int(d), "share": round(float(a), 4), "stars": round(float(b), 3)}
                       for d, a, b in zip(s["days"], s["share7"], s["stars7"])],
        })

    data = {
        "reviews": int(len(corpus)),
        "events": events,
        "dishes": dish_story(corpus),
        **{k: summary[k] for k in ("evaluation_test", "test_reviews", "kappa", "cascade", "security", "impact", "taxonomy", "costs")},
        "alarm_validation": summary["alarm"]["validation"],
        "alarm_main": summary["alarm"]["alarms"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=int), encoding="utf-8")
    print(f"-> {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
