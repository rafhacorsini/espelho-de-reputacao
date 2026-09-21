"""Lista os modelos da OpenAI que a sua chave consegue usar (sem imprimir a chave)."""

import os

from dotenv import load_dotenv
from openai import OpenAI

PREFIXES = ("gpt-5", "gpt-6", "gpt-4o-mini", "text-embedding")


def main() -> None:
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
    if not key:
        raise SystemExit("Nenhuma chave encontrada. Confira o arquivo .env.")

    client = OpenAI(api_key=key)
    ids = sorted(m.id for m in client.models.list())
    print(f"{len(ids)} modelos acessíveis. Os que nos interessam:")
    for model_id in ids:
        if model_id.startswith(PREFIXES):
            print(" ", model_id)


if __name__ == "__main__":
    main()
