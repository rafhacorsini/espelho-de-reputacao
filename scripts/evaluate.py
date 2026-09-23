"""Avalia as previsões contra o rótulo ouro e compara com o baseline de palavras-chave.

Uso:
    uv run python scripts/evaluate.py --split dev --version extract_v1
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from espelho.baseline import extract_baseline
from espelho.metrics import evaluate, polarity_confusion
from espelho.schema import ASPECTS

PILOT = Path("data/synthetic/pilot_reviews.parquet")
REPORTS = Path("reports")


def fmt(value) -> str:
    return "  -  " if value is None else f"{value:.2f}"


def print_table(name: str, result: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"{'aspecto':<20} {'prec':>5} {'recall':>6} {'F1':>5} {'suporte':>8}")
    for aspect in ASPECTS:
        r = result["per_aspect"][aspect]
        print(f"{aspect:<20} {fmt(r['precision']):>5} {fmt(r['recall']):>6} {fmt(r['f1']):>5} {r['support']:>8}")
    micro = result["micro"]
    print(f"{'MICRO':<20} {fmt(micro['precision']):>5} {fmt(micro['recall']):>6} {fmt(micro['f1']):>5} {micro['support']:>8}")
    print(f"{'MACRO F1':<20} {'':>5} {'':>6} {fmt(result['macro_f1']):>5}")
    print(
        f"Reviews sem aspecto no ouro: {result['empty_correct']}/{result['empty_reviews']} "
        "ficaram vazias na previsão (não inventou nada)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--version", default="extract_v1")
    parser.add_argument("--gold", default=None, help="arquivo do gabarito, ex.: dev_v2 (padrão: o próprio split)")
    args = parser.parse_args()

    gold_name = args.gold or args.split
    gold_df = pd.read_parquet(f"data/gold/{gold_name}.parquet")
    pred_df = pd.read_parquet(f"data/predictions/{args.split}_{args.version}.parquet")
    tones = pd.read_parquet(PILOT).set_index("review_id")["tone"]

    gold = {r.review_id: list(r.gold_mentions) for r in gold_df.itertuples()}
    llm = {r.review_id: list(r.pred_mentions) for r in pred_df.itertuples()}
    baseline = {r.review_id: extract_baseline(r.text) for r in gold_df.itertuples()}

    results = {"llm": evaluate(gold, llm), "baseline": evaluate(gold, baseline)}
    print_table(f"LLM ({args.version}) contra o gabarito {gold_name}", results["llm"])
    print_table(f"Baseline de palavras-chave contra o gabarito {gold_name}", results["baseline"])

    print("\n=== Polaridade (entre aspectos que os dois acharam): ouro -> previsto ===")
    for (g, p), n in sorted(polarity_confusion(gold, llm).items()):
        print(f"  {g:>9} -> {p:<9} {n}")

    ironic = [rid for rid in gold if tones.get(rid) == "irônico"]
    if ironic:
        print(f"\n=== Subgrupo: reviews irônicas ({len(ironic)}) ===")
        for name, preds in (("LLM", llm), ("baseline", baseline)):
            sub = evaluate({r: gold[r] for r in ironic}, {r: preds.get(r, []) for r in ironic})
            print(f"  {name:<9} micro F1: {fmt(sub['micro']['f1'])}")

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"eval_{gold_name}_{args.version}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nSalvo em {out}")


if __name__ == "__main__":
    main()
