"""Teste de fumaça: uma chamada à OpenAI que imprime tokens, custo e latência."""

import os
import time

import openai
from dotenv import load_dotenv
from openai import OpenAI

from espelho.cost import call_cost

MODEL = "gpt-5.6-luna"


def main() -> None:
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
    if not key:
        raise SystemExit("Nenhuma chave encontrada. Cole a chave no arquivo .env.")

    client = OpenAI(api_key=key)

    start = time.perf_counter()
    try:
        response = client.responses.create(
            model=MODEL,
            input="Explique em uma frase curta o que é um token em IA.",
            max_output_tokens=400,
            reasoning={"effort": "low"},
        )
    except openai.AuthenticationError:
        raise SystemExit("Chave inválida. Confira o arquivo .env.")
    except openai.APIStatusError as error:
        raise SystemExit(f"A API recusou o pedido ({error.status_code}): {error.message}")
    latency = time.perf_counter() - start

    print(response.output_text)

    usage = response.usage
    details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
    cost = call_cost(MODEL, usage.input_tokens, usage.output_tokens)

    print(f"\nmodelo: {MODEL}")
    print(f"tokens de entrada: {usage.input_tokens}")
    print(
        f"tokens de saída: {usage.output_tokens} "
        f"(dos quais {reasoning_tokens} de raciocínio)"
    )
    print(f"custo estimado: US$ {cost:.6f}")
    print(f"latência: {latency:.2f}s")


if __name__ == "__main__":
    main()
