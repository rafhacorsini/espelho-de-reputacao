"""Importa as reviews reais copiadas à mão (Dia 7). Tudo fica em data/raw/, fora do git.

Uso:
    uv run python scripts/import_real_reviews.py
"""

from pathlib import Path

import pandas as pd

SOURCE = Path("data/raw/reviews_reais.txt")
OUT = Path("data/raw/real_reviews.parquet")
CHANNELS = {"salao": "salao", "salão": "salao", "delivery": "delivery", "?": None}


def parse(text: str) -> tuple[list[dict], list[str]]:
    rows, problems, restaurant = [], [], None
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("==") and line.endswith("=="):
            restaurant = line.strip("= ").replace("Restaurante", "").strip()
            continue
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) != 3:
            problems.append(f"linha {number}: faltam os separadores |")
            continue
        stars, channel, review = parts
        if stars not in {"1", "2", "3", "4", "5"}:
            problems.append(f"linha {number}: estrelas precisam ser de 1 a 5")
            continue
        if channel.lower() not in CHANNELS:
            problems.append(f"linha {number}: canal precisa ser salao, delivery ou ?")
            continue
        if restaurant is None:
            problems.append(f"linha {number}: falta a linha == Restaurante X == antes")
            continue
        if len(review) < 5:
            problems.append(f"linha {number}: texto curto demais")
            continue
        rows.append({"restaurant": restaurant, "stars": int(stars), "channel": CHANNELS[channel.lower()], "text": review})
    return rows, problems


def main() -> None:
    rows, problems = parse(SOURCE.read_text(encoding="utf-8"))
    for p in problems:
        print("  problema:", p)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Nenhuma review válida ainda.")
    df.insert(0, "review_id", [f"REAL{i:03d}" for i in range(1, len(df) + 1)])
    df.to_parquet(OUT, index=False)
    print(f"\n{len(df)} reviews -> {OUT}")
    print("Por restaurante:", df["restaurant"].value_counts().to_dict())
    print("Por nota:", df["stars"].value_counts().sort_index().to_dict())
    print("Por canal:", df["channel"].fillna("?").value_counts().to_dict())


if __name__ == "__main__":
    main()
