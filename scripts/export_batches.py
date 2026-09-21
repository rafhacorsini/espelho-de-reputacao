"""Prepara os lotes de prompt para gerar o texto das reviews num chat gratuito.

Uso:
    uv run python scripts/export_batches.py              # 180 reviews, lotes de 30
    uv run python scripts/export_batches.py --retry      # só o que faltou na importação
"""

import argparse
from pathlib import Path

import pandas as pd

from espelho.synth.prompts import build_prompt, select_pilot

TRUTH = Path("data/synthetic/truth.parquet")
PILOT = Path("data/synthetic/pilot_truth.parquet")
MISSING = Path("data/synthetic/missing_ids.txt")
BATCH_DIR = Path("data/synthetic/prompt_batches")
TEXTS_DIR = Path("data/synthetic/texts")


def write_batches(df: pd.DataFrame, batch_size: int, prefix: str) -> list[Path]:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start : start + batch_size]
        path = BATCH_DIR / f"{prefix}_{start // batch_size + 1:02d}.txt"
        path.write_text(build_prompt(chunk), encoding="utf-8")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=180, help="tamanho da amostra piloto")
    parser.add_argument("--batch-size", type=int, default=30, help="reviews por lote")
    parser.add_argument("--retry", action="store_true", help="exporta só os ids que faltaram")
    args = parser.parse_args()

    TEXTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.retry:
        pilot = pd.read_parquet(PILOT)
        missing = MISSING.read_text(encoding="utf-8").split()
        df = pilot[pilot["review_id"].isin(missing)].reset_index(drop=True)
        if df.empty:
            print("Nada faltando.")
            return
        paths = write_batches(df, args.batch_size, "retry")
    else:
        truth = pd.read_parquet(TRUTH)
        df = select_pilot(truth, n=args.n)
        df.to_parquet(PILOT, index=False)
        paths = write_batches(df, args.batch_size, "batch")

    print(f"{len(df)} reviews em {len(paths)} lote(s):")
    for path in paths:
        print(f"  {path}")
    print(f"\nCole a resposta de cada lote em {TEXTS_DIR}\\<mesmo nome>.jsonl")


if __name__ == "__main__":
    main()
