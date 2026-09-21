"""Teste de fumaça: uma chamada ao Haiku 4.5 que imprime tokens, custo e latência."""

import os
import time

import anthropic
from dotenv import load_dotenv

from espelho.cost import call_cost

MODEL = "claude-haiku-4-5"


def main() -> None:
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY está vazia. Cole a chave no arquivo .env e rode de novo."
        )

    client = anthropic.Anthropic()  # lê ANTHROPIC_API_KEY do ambiente

    start = time.perf_counter()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "Explique em uma frase curta o que é um token em IA.",
                }
            ],
        )
    except anthropic.AuthenticationError:
        raise SystemExit("Chave inválida. Confira o ANTHROPIC_API_KEY no arquivo .env.")
    latency = time.perf_counter() - start

    for block in response.content:
        if block.type == "text":
            print(block.text)

    usage = response.usage
    cost = call_cost(MODEL, usage.input_tokens, usage.output_tokens)
    print(f"\nmodelo: {MODEL}")
    print(
        f"tokens de entrada: {usage.input_tokens} | tokens de saída: {usage.output_tokens}"
    )
    print(f"custo estimado: US$ {cost:.6f}")
    print(f"latência: {latency:.2f}s")


if __name__ == "__main__":
    main()
