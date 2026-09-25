"""Roda o extrator (extract_v2) em todas as reviews do corpus, várias ao mesmo tempo.

Pode ser interrompido e rodado de novo: continua de onde parou.

Uso:
    uv run python scripts/extract_corpus.py
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from espelho.extract import extract_one

MODEL = "gpt-5.6-luna"
VERSION = "extract_v2"
WORKERS = 8

CORPUS = Path("data/synthetic/corpus_reviews.parquet")
PARTIAL = Path("data/predictions/corpus_extract_v2.jsonl")
OUT = Path("data/predictions/corpus_extract_v2.parquet")


def main() -> None:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY"), max_retries=5)

    corpus = pd.read_parquet(CORPUS)
    PARTIAL.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if PARTIAL.exists():
        done = {json.loads(line)["review_id"] for line in PARTIAL.read_text(encoding="utf-8").splitlines() if line}
    todo = corpus[~corpus["review_id"].isin(done)]
    print(f"{len(done)} já feitas, {len(todo)} a fazer")

    def run(row):
        result = extract_one(client, MODEL, row.text, channel=row.channel, version=VERSION)
        return row.review_id, result

    total_cost, failures, started = 0.0, 0, time.perf_counter()
    with ThreadPoolExecutor(WORKERS) as pool, PARTIAL.open("a", encoding="utf-8") as out:
        futures = [pool.submit(run, row) for row in todo.itertuples()]
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                review_id, result = future.result()
            except Exception as error:  # a review que falhou fica para a próxima execução
                failures += 1
                print(f"  falhou: {error}")
                continue
            total_cost += result["cost"]
            out.write(json.dumps({"review_id": review_id, **result}, ensure_ascii=False) + "\n")
            if i % 250 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} | custo acumulado: US$ {total_cost:.4f}")

    records = [json.loads(line) for line in PARTIAL.read_text(encoding="utf-8").splitlines() if line]
    df = pd.DataFrame(records).drop_duplicates("review_id", keep="last")
    df = df.rename(columns={"mentions": "pred_mentions"})
    df.to_parquet(OUT, index=False)

    elapsed = time.perf_counter() - started
    print(f"\n{len(df)} de {len(corpus)} reviews extraídas -> {OUT} ({failures} falhas)")
    print(f"Custo desta execução: US$ {total_cost:.4f} | tempo: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
