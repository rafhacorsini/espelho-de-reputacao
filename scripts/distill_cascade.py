"""Dia 6, parte A: aluno destilado e cascata aluno -> LLM.

- Professor: as previsões do extract_v2 no corpus.
- Treino do aluno: todas as reviews do corpus MENOS as 180 do piloto (de onde
  saem o dev e o test), para não haver vazamento.
- Regra de escolha do limiar, fixada antes: o mais barato com F1 no dev até
  2 pontos abaixo do LLM puro. O test é só para reportar.

Uso:
    uv run python scripts/distill_cascade.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from espelho.cascade import cascade, student_predict, train_student
from espelho.metrics import evaluate
from espelho.taxonomy import embed

CORPUS = Path("data/synthetic/corpus_reviews.parquet")
TEACHER = Path("data/predictions/corpus_extract_v2.parquet")
TEACHER_LOG = Path("data/predictions/corpus_extract_v2.jsonl")
PILOT = Path("data/synthetic/pilot_reviews.parquet")
EMB = Path("data/cascade/review_embeddings.npy")
RESULTS = Path("reports/cascade_results.json")

TAUS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
MAX_F1_LOSS = 0.02


def split_eval(name: str) -> tuple[dict, dict]:
    gold_df = pd.read_parquet(f"data/gold/{name}_v2.parquet")
    llm_df = pd.read_parquet(f"data/predictions/{name}_extract_v2.parquet")
    gold = {r.review_id: list(r.gold_mentions) for r in gold_df.itertuples()}
    llm = {r.review_id: list(r.pred_mentions) for r in llm_df.itertuples()}
    return gold, llm


def main() -> None:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY"), max_retries=5)

    corpus = pd.read_parquet(CORPUS).merge(pd.read_parquet(TEACHER)[["review_id", "pred_mentions"]], on="review_id")
    EMB.parent.mkdir(parents=True, exist_ok=True)
    if EMB.exists() and len(np.load(EMB)) == len(corpus):
        X = np.load(EMB)
        emb_cost = 0.0
    else:
        X, emb_cost = embed(client, corpus["text"].tolist())
        np.save(EMB, X)
    per_review_emb_cost = 0.02 * 45 / 1_000_000  # ~45 tokens por review, US$ 0,02 por milhão
    print(f"Embeddings das {len(corpus)} reviews: custo desta execução US$ {emb_cost:.4f}")

    log = [json.loads(line) for line in TEACHER_LOG.read_text(encoding="utf-8").splitlines() if line]
    llm_cost = float(np.mean([r["cost"] for r in log]))
    llm_latency = float(np.mean([r["latency_s"] for r in log]))
    print(f"LLM medido: US$ {llm_cost:.6f} e {llm_latency:.1f}s por review")

    pilot_ids = set(pd.read_parquet(PILOT)["review_id"])
    train = ~corpus["review_id"].isin(pilot_ids).to_numpy()
    models = train_student(X[train], corpus.loc[train, "pred_mentions"].tolist())
    print(f"Aluno treinado com {train.sum()} reviews rotuladas pelo professor (sem as 180 do piloto)")

    index = {rid: i for i, rid in enumerate(corpus["review_id"])}
    results = {}
    for name in ("dev", "test"):
        gold, llm = split_eval(name)
        ids = list(gold)
        student, confidence = student_predict(models, X[[index[r] for r in ids]])
        llm_list = [llm[r] for r in ids]
        rows = []
        for tau in TAUS:
            merged, share = cascade(student, llm_list, confidence, tau)
            f1 = evaluate(gold, dict(zip(ids, merged)))["micro"]["f1"]
            rows.append({"tau": tau, "f1": f1, "share_llm": share, "cost_per_1000": 1000 * (share * llm_cost + per_review_emb_cost)})
        results[name] = {
            "llm_f1": evaluate(gold, llm)["micro"]["f1"],
            "student_f1": evaluate(gold, dict(zip(ids, student)))["micro"]["f1"],
            "sweep": rows,
        }

    dev = results["dev"]
    ok = [r for r in dev["sweep"] if r["f1"] >= dev["llm_f1"] - MAX_F1_LOSS]
    chosen = min(ok, key=lambda r: r["cost_per_1000"])["tau"] if ok else None

    print("\n=== Varredura do limiar no dev (o aluno responde quando está seguro) ===")
    print(f"{'limiar':>6} {'F1':>5} {'vai ao LLM':>10} {'US$/1.000':>10}")
    for r in dev["sweep"]:
        mark = "  <- escolhido" if r["tau"] == chosen else ""
        print(f"{r['tau']:>6.2f} {r['f1']:>5.2f} {r['share_llm']:>10.0%} {r['cost_per_1000']:>10.3f}{mark}")

    test = results["test"]
    test_choice = next(r for r in test["sweep"] if r["tau"] == chosen) if chosen is not None else None
    llm_per_1000 = 1000 * llm_cost
    student_per_1000 = 1000 * per_review_emb_cost
    print("\n=== Resultado final no test ===")
    print(f"{'sistema':<24} {'F1 test':>7} {'US$/1.000':>10}")
    print(f"{'LLM puro':<24} {test['llm_f1']:>7.2f} {llm_per_1000:>10.3f}")
    print(f"{'Aluno sozinho':<24} {test['student_f1']:>7.2f} {student_per_1000:>10.3f}")
    if test_choice:
        print(f"{'Cascata (limiar ' + str(chosen) + ')':<24} {test_choice['f1']:>7.2f} {test_choice['cost_per_1000']:>10.3f}"
              f"   ({test_choice['share_llm']:.0%} foi ao LLM)")

    RESULTS.write_text(
        json.dumps({"llm_cost_per_review": llm_cost, "llm_latency_s": llm_latency, "chosen_tau": chosen, **results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n-> {RESULTS}")


if __name__ == "__main__":
    main()
