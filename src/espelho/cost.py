"""Preços por 1 milhão de tokens (US$) e cálculo do custo de uma chamada.

Fonte: página oficial de preços da OpenAI, lida em 21/09/2026. Os preços mudam,
então confira antes de confiar nos números.

Atenção: nos modelos de raciocínio (família GPT-5 em diante) os tokens de
raciocínio são cobrados como saída. `output_tokens` da API já os inclui.
"""

PRICES_PER_MTOK = {
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00},
    "gpt-5.6-sol": {"input": 4.00, "output": 20.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
}

# O Batch API cobra metade do preço padrão, na entrada e na saída.
BATCH_DISCOUNT = 0.5


def call_cost(
    model: str, input_tokens: int, output_tokens: int, batch: bool = False
) -> float:
    """Custo em US$ de uma chamada: entrada e saída são cobradas separadamente."""
    prices = PRICES_PER_MTOK[model]
    cost = (
        input_tokens * prices["input"] + output_tokens * prices["output"]
    ) / 1_000_000
    return cost * BATCH_DISCOUNT if batch else cost
