"""Roda a extração num conjunto inteiro (dev ou test), registrando cada chamada.

Uso:
    uv run python scripts/extract_batch.py --split dev
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from espelho.extract import PROMPT_VERSION, PROMPT_VERSIONS, extract_one

MODEL = "gpt-5.6-luna"
LOG_PATH = Path("logs/extract_calls.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--version", choices=PROMPT_VERSIONS, default=PROMPT_VERSION)
    args = parser.parse_args()

    if args.split == "test":
        raise SystemExit(
            "O test só abre no Dia 3, uma vez. Se você quer mesmo abrir agora, "
            "edite este script — não vou abrir por engano."
        )

    load_dotenv()
    key = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
    client = OpenAI(api_key=key)

    df = pd.read_parquet(f"data/gold/{args.split}.parquet")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    predictions, total_cost = [], 0.0
    started = time.perf_counter()
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            result = extract_one(client, MODEL, row["text"], channel=row["channel"], version=args.version)
            total_cost += result["cost"]

            log_file.write(
                json.dumps(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "review_id": row["review_id"],
                        "split": args.split,
                        **{k: v for k, v in result.items() if k != "mentions"},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            predictions.append({"review_id": row["review_id"], "pred_mentions": result["mentions"]})

            if i % 20 == 0 or i == len(df):
                print(f"{i}/{len(df)} | custo acumulado: US$ {total_cost:.4f}")

    out = Path(f"data/predictions/{args.split}_{args.version}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_parquet(out, index=False)

    elapsed = time.perf_counter() - started
    print(f"\nSalvo em {out}")
    print(f"Custo total: US$ {total_cost:.4f} | tempo: {elapsed:.1f}s | modelo: {MODEL}")


if __name__ == "__main__":
    main()
