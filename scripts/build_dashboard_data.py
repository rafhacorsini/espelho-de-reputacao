"""Junta os números de todos os dias num arquivo só, que o dashboard lê.

O dashboard nunca chama a API: tudo aqui é pré-calculado e versionado.

Uso:
    uv run python scripts/build_dashboard_data.py
"""

import json
from pathlib import Path

import pandas as pd

from espelho.baseline import extract_baseline
from espelho.metrics import bootstrap_f1

OUT = Path("reports/dashboard_summary.json")

# Custos de API medidos em cada etapa (US$), anotados quando cada uma rodou.
MEASURED_COSTS = [
    ("Dia 0", "teste de fumaça", 0.00004),
    ("Dia 1", "correções de texto pela API", 0.0045),
    ("Dia 2", "extração no dev (v1)", 0.0182),
    ("Dia 3", "v2 no dev + test (v1 e v2)", 0.0418),
    ("Dia 4", "texto do corpus + extração nas 3.555 + embeddings + nomes", 1.0005),
    ("Dia 5", "estatística local", 0.0),
    ("Dia 6", "embeddings das reviews + ataques + v3 no dev", 0.0864),
]


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    test_gold_df = pd.read_parquet("data/gold/test_v2.parquet")
    gold = {r.review_id: list(r.gold_mentions) for r in test_gold_df.itertuples()}
    systems = {
        "LLM extract_v2 (oficial)": "data/predictions/test_extract_v2.parquet",
        "LLM extract_v1": "data/predictions/test_extract_v1.parquet",
    }
    evaluation = []
    for name, path in systems.items():
        pred = {r.review_id: list(r.pred_mentions) for r in pd.read_parquet(path).itertuples()}
        evaluation.append({"system": name, "f1": bootstrap_f1(gold, pred)})
    base = {r.review_id: extract_baseline(r.text) for r in test_gold_df.itertuples()}
    evaluation.append({"system": "Baseline de palavras-chave", "f1": bootstrap_f1(gold, base)})

    adjudication = load("data/gold/test_v2_adjudication.json")
    cascade = load("reports/cascade_results.json")
    chosen = next(r for r in cascade["test"]["sweep"] if r["tau"] == cascade["chosen_tau"])
    security = load("reports/security_results.json")
    day5 = load("reports/day5_results.json")
    taxonomy = load("data/taxonomy/taxonomy_draft.json")

    summary = {
        "evaluation_test": evaluation,
        "test_reviews": len(gold),
        "kappa": adjudication["agreement_before_adjudication"]["kappa"],
        "cascade": [
            {"system": "LLM puro", "f1_test": cascade["test"]["llm_f1"], "cost_per_1000": 1000 * cascade["llm_cost_per_review"], "share_llm": 1.0},
            {"system": "Aluno sozinho", "f1_test": cascade["test"]["student_f1"], "cost_per_1000": 1000 * 0.02 * 45 / 1_000_000, "share_llm": 0.0},
            {"system": f"Cascata (limiar {str(cascade['chosen_tau']).replace('.', ',')})", "f1_test": chosen["f1"], "cost_per_1000": chosen["cost_per_1000"], "share_llm": chosen["share_llm"]},
        ],
        "security": security,
        "alarm": {k: day5[k] for k in ("alarms", "star_alarms", "validation", "sensitivity")},
        "impact": day5["impact"],
        "taxonomy": [{k: c[k] for k in ("cluster", "size", "name", "polarity", "parent_aspect")} for c in taxonomy],
        "costs": [{"day": d, "what": w, "usd": c} for d, w, c in MEASURED_COSTS],
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"-> {OUT} | custo total medido: US$ {sum(c for *_, c in MEASURED_COSTS):.2f}")


if __name__ == "__main__":
    main()
