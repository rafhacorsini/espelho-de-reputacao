"""Gera pela API o texto de todas as reviews que ainda não têm texto (Dia 4).

As 180 do piloto já têm texto; aqui geramos as outras, em lotes de 30, várias
chamadas ao mesmo tempo. O que faltar volta para uma nova rodada.

Uso:
    uv run python scripts/generate_corpus.py --limit 2    # teste: só 2 lotes
    uv run python scripts/generate_corpus.py              # tudo
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from espelho.cost import call_cost
from espelho.synth.prompts import build_prompt
from espelho.synth.texts import parse_lines, validate

MODEL = "gpt-5.6-luna"
BATCH_SIZE = 30
WORKERS = 8
MAX_ROUNDS = 3

TRUTH = Path("data/synthetic/truth.parquet")
PILOT = Path("data/synthetic/pilot_reviews.parquet")
GENERATED = Path("data/synthetic/generated_texts.jsonl")
CORPUS = Path("data/synthetic/corpus_reviews.parquet")


def generate_chunk(client: OpenAI, chunk: pd.DataFrame) -> tuple[list[dict], float]:
    response = client.responses.create(
        model=MODEL,
        input=build_prompt(chunk),
        max_output_tokens=6000,
        reasoning={"effort": "low"},
    )
    records, _ = parse_lines(response.output_text)
    valid, _ = validate(records, chunk["review_id"], allow_duplicate_text=True)
    return valid, call_cost(MODEL, response.usage.input_tokens, response.usage.output_tokens)


def load_generated() -> dict:
    if not GENERATED.exists():
        return {}
    records = [json.loads(line) for line in GENERATED.read_text(encoding="utf-8").splitlines() if line]
    return {r["review_id"]: r["text"] for r in records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="só N lotes (para medir custo)")
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY"), max_retries=5)

    truth = pd.read_parquet(TRUTH)
    pilot = pd.read_parquet(PILOT)
    total_cost, started = 0.0, time.perf_counter()

    for round_number in range(1, MAX_ROUNDS + 1):
        done = set(pilot["review_id"]) | set(load_generated())
        todo = truth[~truth["review_id"].isin(done)].reset_index(drop=True)
        if todo.empty:
            break
        chunks = [todo.iloc[i : i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
        if args.limit:
            chunks = chunks[: args.limit]
        print(f"Rodada {round_number}: {len(todo)} reviews sem texto, {len(chunks)} lote(s)")

        with ThreadPoolExecutor(WORKERS) as pool, GENERATED.open("a", encoding="utf-8") as out:
            futures = [pool.submit(generate_chunk, client, chunk) for chunk in chunks]
            for i, future in enumerate(as_completed(futures), start=1):
                try:
                    valid, cost = future.result()
                except Exception as error:  # um lote que falhou volta na próxima rodada
                    print(f"  lote falhou: {error}")
                    continue
                total_cost += cost
                for record in valid:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                if i % 10 == 0 or i == len(chunks):
                    print(f"  {i}/{len(chunks)} lotes | custo acumulado: US$ {total_cost:.4f}")
        if args.limit:
            break

    generated = load_generated()
    texts = {**generated, **dict(zip(pilot["review_id"], pilot["text"]))}
    corpus = truth[truth["review_id"].isin(texts)].copy()
    corpus["text"] = corpus["review_id"].map(texts)
    corpus["text_source"] = ["piloto" if rid in set(pilot["review_id"]) else "api" for rid in corpus["review_id"]]
    corpus.to_parquet(CORPUS, index=False)

    elapsed = time.perf_counter() - started
    print(f"\n{len(corpus)} de {len(truth)} reviews com texto -> {CORPUS}")
    print(f"Custo desta execução: US$ {total_cost:.4f} | tempo: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
