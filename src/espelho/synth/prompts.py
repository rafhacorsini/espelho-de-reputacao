"""Prompt para gerar o texto das reviews num chat e seleção da amostra piloto.

O texto pode ser gerado por qualquer IA (inclusive um chat gratuito). O prompt
recebe a verdade sorteada e pede só o texto de volta, em JSON por linha.
"""

import json

import numpy as np
import pandas as pd

from espelho.schema import ASPECTS
from espelho.synth.truth import MAX_MENTIONS

PROMPT_TEMPLATE = """Você vai escrever avaliações de clientes de restaurantes, como as do Google Maps, em português do Brasil. Elas formam um conjunto de dados de teste para um sistema de análise de reviews.

Para CADA linha de entrada (um objeto JSON), escreva UMA avaliação.

CAMPOS DA ENTRADA
- review_id: identificador. Copie na saída, sem alterar.
- channel: "salao" = o cliente comeu no restaurante; "delivery" = pediu para entregar.
- cuisine: tipo de restaurante.
- stars: nota que o cliente deu (1 a 5). O tom geral tem que combinar com ela. Não escreva a nota no texto.
- mentions: os ÚNICOS assuntos que a avaliação pode tratar. Cada item tem "aspect" e "polarity" ("positivo" = elogio, "negativo" = reclamação).
- tone: o jeito de escrever.
- length: o tamanho aproximado.
- typos: se true, inclua 1 ou 2 erros de digitação naturais.
- dish: se preenchido e houver elogio à comida, cite esse prato.

ASSUNTOS (aspect) E O QUE SIGNIFICAM
{aspects}

REGRAS
1. Trate SOMENTE dos assuntos de "mentions", cada um com a polaridade indicada e de forma que um leitor reconheça o assunto. Não fale de nenhum outro assunto da lista acima.
2. Se "mentions" estiver vazio, escreva uma avaliação genérica, sem citar comida, preço, atendimento, espera, entrega, ambiente ou canal de contato. Exemplos: "Gostei, recomendo." ou "Lugar ok.". Combine com a nota.
3. Tom "irônico": diga o contrário do que sente, mas deixe claro para um leitor atento qual é o sentido real e qual é a polaridade.
4. Escreva como cliente de verdade. Nunca use os nomes técnicos dos assuntos (como prazo_entrega).
5. Varie: não repita aberturas, frases nem exemplos entre as avaliações do lote.
6. Não use nomes de pessoas, de marcas ou de restaurantes reais. Sem dados pessoais. Emojis, no máximo em 1 de cada 5 avaliações.

SAÍDA
Responda com exatamente uma linha JSON por avaliação, na mesma ordem e na mesma quantidade da entrada. Sem texto antes ou depois, sem blocos de código e sem numeração. Formato de cada linha:
{"review_id": "R00001", "text": "..."}

EXEMPLOS (só para mostrar o formato; não repita estes textos)
Entrada: {"review_id": "X1", "channel": "delivery", "cuisine": "pizzaria", "stars": 2, "mentions": [{"aspect": "prazo_entrega", "polarity": "negativo"}], "tone": "informal com gíria", "length": "2 a 3 frases", "typos": false, "dish": null}
Saída: {"review_id": "X1", "text": "Pedi às 20h e a pizza só chegou às 22h30, tava todo mundo morrendo de fome. Não peço mais."}
Entrada: {"review_id": "X2", "channel": "salao", "cuisine": "churrascaria", "stars": 4, "mentions": [], "tone": "curto e seco", "length": "1 frase", "typos": false, "dish": null}
Saída: {"review_id": "X2", "text": "Gostei, voltaria."}

ENTRADA (uma linha por avaliação)
{rows}
"""


def _prompt_row(row) -> dict:
    dish = row["dish"]
    return {
        "review_id": row["review_id"],
        "channel": row["channel"],
        "cuisine": row["cuisine"],
        "stars": int(row["stars"]),
        "mentions": json.loads(row["mentions"]),
        "tone": row["tone"],
        "length": row["length_hint"],
        "typos": bool(row["has_typos"]),
        "dish": dish if isinstance(dish, str) else None,
    }


def build_prompt(rows: pd.DataFrame) -> str:
    """Monta o prompt completo para um lote de reviews."""
    aspects = "\n".join(f"- {name}: {meaning}" for name, meaning in ASPECTS.items())
    lines = "\n".join(
        json.dumps(_prompt_row(row), ensure_ascii=False) for _, row in rows.iterrows()
    )
    return PROMPT_TEMPLATE.replace("{aspects}", aspects).replace("{rows}", lines)


def select_pilot(df: pd.DataFrame, n: int = 180, seed: int = 123) -> pd.DataFrame:
    """Escolhe a amostra piloto: casos difíceis primeiro, o resto espalhado por loja.

    É a amostra de onde sai o conjunto ouro que você rotula à mão.
    """
    rng = np.random.default_rng(seed)
    chosen: list[str] = []

    def take(mask, k):
        pool = df[mask & ~df["review_id"].isin(chosen)]
        picked = pool.sample(n=min(k, len(pool)), random_state=int(rng.integers(0, 2**31 - 1)))
        chosen.extend(picked["review_id"])

    take(df["n_mentions"] == 0, 20)  # sem assunto: testa se o sistema inventa
    take(df["sarcastic"] & (df["n_mentions"] > 0), 20)  # ironia
    take(df["n_mentions"] == MAX_MENTIONS, 20)  # vários assuntos na mesma review

    remaining = n - len(chosen)
    per_store = -(-remaining // df["store_id"].nunique())
    for store_id in sorted(df["store_id"].unique()):
        take(df["store_id"] == store_id, per_store)

    pilot = df[df["review_id"].isin(chosen[:n])]
    return pilot.sort_values("review_id").reset_index(drop=True)
