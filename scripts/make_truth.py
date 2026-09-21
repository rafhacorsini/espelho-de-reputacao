"""Gera a verdade sorteada (sem texto) e mostra se os 5 eventos plantados aparecem."""

from pathlib import Path

from espelho.synth.checks import event_check
from espelho.synth.truth import generate_truth

OUTPUT = Path("data/synthetic/truth.parquet")


def main() -> None:
    df = generate_truth()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)

    print(f"{len(df)} reviews salvas em {OUTPUT}\n")
    print("Reviews por loja:")
    print(df.groupby("store_name").size().to_string(), "\n")
    print("Reviews por canal:")
    print(df["channel"].value_counts().to_string(), "\n")
    print(f"Sem nenhuma menção: {(df['n_mentions'] == 0).mean():.1%}")
    print(f"Nota média observada: {df['stars'].mean():.2f}\n")
    print("Eventos plantados (taxa antes e depois do dia de início):")
    print(event_check(df).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
