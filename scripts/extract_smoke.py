"""Passo 4 do Dia 2: chama a extração numa única review, pra ler com os olhos
antes de automatizar em lote.

Uso:
    uv run python scripts/extract_smoke.py           # a 1ª review do dev
    uv run python scripts/extract_smoke.py --n 10     # as 10 primeiras
"""

import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from espelho.extract import extract_one

MODEL = "gpt-5.6-luna"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()

    load_dotenv()
    key = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
    client = OpenAI(api_key=key)

    dev = pd.read_parquet("data/gold/dev.parquet").head(args.n)

    total_cost = 0.0
    for _, row in dev.iterrows():
        result = extract_one(client, MODEL, row["text"])
        total_cost += result["cost"]

        print(f"\n{row['review_id']} ({row['stars']}★): {row['text']}")
        print(f"  rótulo ouro: {row['gold_mentions']}")
        if result["mentions"]:
            for m in result["mentions"]:
                flag = "" if m["evidence_verified"] else "  <-- EVIDÊNCIA NÃO BATE COM O TEXTO"
                print(f"  extraído: {m['aspect']} ({m['polarity']}) \"{m['evidencia']}\"{flag}")
        else:
            print("  extraído: (nenhuma menção)")
        print(
            f"  tokens: {result['input_tokens']} in / {result['output_tokens']} out | "
            f"custo: US$ {result['cost']:.6f} | latência: {result['latency_s']}s"
        )

    print(f"\n--- Custo total: US$ {total_cost:.6f} para {len(dev)} review(s) ---")
    print(f"Extrapolado para 100 (dev): US$ {total_cost / len(dev) * 100:.4f}")


if __name__ == "__main__":
    main()
