"""Gera texto de reviews chamando a API diretamente (para lotes pequenos, como retries).

Uso:
    uv run python scripts/generate_via_api.py data/synthetic/prompt_batches/retry_01.txt
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from espelho.cost import call_cost

MODEL = "gpt-5.6-luna"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt_file", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    load_dotenv()
    key = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
    client = OpenAI(api_key=key)

    prompt = args.prompt_file.read_text(encoding="utf-8")
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        max_output_tokens=4000,
        reasoning={"effort": "low"},
    )

    out = args.out or Path("data/synthetic/texts") / (args.prompt_file.stem + ".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(response.output_text, encoding="utf-8")

    usage = response.usage
    cost = call_cost(MODEL, usage.input_tokens, usage.output_tokens)
    print(f"Salvo em {out}")
    print(
        f"tokens entrada: {usage.input_tokens} | saída: {usage.output_tokens} | "
        f"custo estimado: US$ {cost:.6f}"
    )


if __name__ == "__main__":
    main()
