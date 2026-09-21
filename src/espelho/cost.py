"""Preços por 1 milhão de tokens (US$) e cálculo do custo de uma chamada.

Fonte: tabela da documentação da Claude API (cache de jun/2026). Os preços podem
ter mudado, então confira antes de confiar nos números.
"""

PRICES_PER_MTOK = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
}

# O Batch API cobra metade do preço padrão.
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
