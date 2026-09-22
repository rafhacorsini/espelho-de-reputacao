"""Importa os textos colados do chat, valida e junta com a verdade sorteada.

Uso:
    uv run python scripts/import_texts.py
"""

from pathlib import Path

import pandas as pd

from espelho.synth.texts import parse_lines, validate

PILOT = Path("data/synthetic/pilot_truth.parquet")
TEXTS_DIR = Path("data/synthetic/texts")
OUTPUT = Path("data/synthetic/pilot_reviews.parquet")
MISSING = Path("data/synthetic/missing_ids.txt")


def main() -> None:
    pilot = pd.read_parquet(PILOT)

    files = sorted(p for p in TEXTS_DIR.glob("*") if p.name.endswith((".jsonl", ".jsonl.txt", ".json")))
    if not files:
        raise SystemExit(f"Nenhum arquivo em {TEXTS_DIR}. Cole as respostas do chat lá.")

    records = []
    for path in files:
        content = path.read_text(encoding="utf-8-sig")
        if content.lstrip().startswith("Você vai escrever avaliações"):
            print(f"{path.name}: parece o PROMPT, não a resposta da IA. Pulei este arquivo.")
            continue
        parsed, errors = parse_lines(content)
        records.extend(parsed)
        print(f"{path.name}: {len(parsed)} linhas lidas, {len(errors)} com erro")
        for number, reason in errors[:5]:
            print(f"    linha {number}: {reason}")

    valid, problems = validate(records, pilot["review_id"])
    merged = pilot.merge(pd.DataFrame(valid), on="review_id", how="inner")
    merged.to_parquet(OUTPUT, index=False)

    missing = sorted(set(pilot["review_id"]) - {r["review_id"] for r in valid})
    MISSING.write_text("\n".join(missing), encoding="utf-8")

    print(f"\nVálidos: {len(valid)} de {len(pilot)}. Salvos em {OUTPUT}")
    if problems:
        print(f"Problemas ({len(problems)}):")
        for review_id, reason in problems[:15]:
            print(f"  {review_id}: {reason}")
    if missing:
        print(f"\nFaltam {len(missing)} reviews. Rode: uv run python scripts/export_batches.py --retry")


if __name__ == "__main__":
    main()
