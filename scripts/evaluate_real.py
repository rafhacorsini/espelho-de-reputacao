"""Mede o sistema nas reviews reais (Dia 7). Roda o extract_v2 UMA vez.

Privacidade: previsões e erros ficam em data/raw/ (fora do git); o relatório
público (reports/real_results.json) só tem números agregados.

Uso:
    uv run python scripts/evaluate_real.py
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from espelho.baseline import extract_baseline
from espelho.extract import extract_one
from espelho.gold import load_labels
from espelho.metrics import as_tuples, bootstrap_f1, evaluate

MODEL, VERSION = "gpt-5.6-luna", "extract_v2"
REVIEWS = Path("data/raw/real_reviews.parquet")
LABELS = Path("data/raw/real_labels.jsonl")
PREDICTIONS = Path("data/raw/real_predictions.parquet")
ERRORS = Path("data/raw/real_errors.md")
PUBLIC = Path("reports/real_results.json")
SYNTHETIC_TEST_F1 = (0.94, 0.90, 0.99)


def main() -> None:
    reviews = pd.read_parquet(REVIEWS)
    labels = {rid: r["mentions"] for rid, r in load_labels(LABELS).items() if not r.get("discarded")}
    missing = set(reviews["review_id"]) - set(load_labels(LABELS))
    if missing:
        raise SystemExit(f"Rotule todas antes de o modelo ver. Faltam {len(missing)}.")
    reviews = reviews[reviews["review_id"].isin(labels)]

    if PREDICTIONS.exists():
        preds_df = pd.read_parquet(PREDICTIONS)
        print("Previsões já existem: o modelo já viu estas reviews. Só recalculando as métricas.")
    else:
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY"), max_retries=5)

        def run(row):
            result = extract_one(client, MODEL, row.text, channel=row.channel, version=VERSION)
            return {"review_id": row.review_id, "pred_mentions": result["mentions"], "cost": result["cost"]}

        with ThreadPoolExecutor(8) as pool:
            preds_df = pd.DataFrame(list(pool.map(run, reviews.itertuples())))
        preds_df.to_parquet(PREDICTIONS, index=False)
        print(f"Custo: US$ {preds_df['cost'].sum():.4f}")

    llm = {r.review_id: list(r.pred_mentions) for r in preds_df.itertuples()}
    baseline = {r.review_id: extract_baseline(r.text) for r in reviews.itertuples()}
    llm_f1 = bootstrap_f1(labels, llm)
    base_f1 = bootstrap_f1(labels, baseline)
    result = evaluate(labels, llm)

    print(f"\n{len(labels)} reviews reais, {result['micro']['support']} menções no seu gabarito")
    print(f"{'sistema':<22} {'F1':>5}  intervalo 95%")
    print(f"{'LLM (extract_v2)':<22} {llm_f1[0]:>5.2f}  [{llm_f1[1]:.2f} a {llm_f1[2]:.2f}]")
    print(f"{'Baseline':<22} {base_f1[0]:>5.2f}  [{base_f1[1]:.2f} a {base_f1[2]:.2f}]")
    print(f"{'(test sintético, LLM)':<22} {SYNTHETIC_TEST_F1[0]:>5.2f}  [{SYNTHETIC_TEST_F1[1]:.2f} a {SYNTHETIC_TEST_F1[2]:.2f}]")
    print(f"Precisão {result['micro']['precision']:.2f} | recall {result['micro']['recall']:.2f} | "
          f"reviews sem aspecto que ficaram vazias: {result['empty_correct']}/{result['empty_reviews']}")

    text_by_id = dict(zip(reviews["review_id"], reviews["text"]))
    lines = ["# Erros nas reviews reais (PRIVADO, não publicar)", ""]
    for rid, gold in labels.items():
        g, p = as_tuples(gold), as_tuples(llm.get(rid, []))
        if g != p:
            lines += [f"## {rid}", f"> {text_by_id[rid]}", f"- inventou: {sorted(p - g)}", f"- perdeu: {sorted(g - p)}", ""]
    ERRORS.write_text("\n".join(lines), encoding="utf-8")
    print(f"Erros para a análise (privado): {ERRORS}")

    PUBLIC.write_text(json.dumps({
        "reviews": len(labels), "restaurants": int(reviews["restaurant"].nunique()),
        "llm_f1": llm_f1, "baseline_f1": base_f1, "synthetic_test_f1": SYNTHETIC_TEST_F1,
        "precision": result["micro"]["precision"], "recall": result["micro"]["recall"],
        "per_aspect_f1": {a: r["f1"] for a, r in result["per_aspect"].items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Números agregados (público): {PUBLIC}")


if __name__ == "__main__":
    main()
