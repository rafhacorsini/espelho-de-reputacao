"""Extração estruturada: transforma o texto de uma review numa lista de
(aspecto, polaridade, evidência), usando a Responses API com saída validada.

A evidência tem que ser cópia literal de um trecho da review. Depois da
chamada, conferimos isso em código — é o nosso detector barato de alucinação.
"""

import time
from enum import StrEnum
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from espelho.cost import call_cost
from espelho.schema import ASPECTS

# Um só lugar de verdade: os aspectos vêm de schema.py, nunca duplicados aqui.
Aspect = StrEnum("Aspect", {name: name for name in ASPECTS})

PROMPT_VERSIONS = ("extract_v1", "extract_v2")
PROMPT_VERSION = "extract_v2"

# Regras que o v2 acrescenta, vindas das decisões do gabarito v2 (Dia 3).
V2_RULES = """7. A mensagem informa o canal: "salao" (comeu no restaurante) ou "delivery" (pediu para entregar). Rapidez ou demora sem dizer onde foi: se o canal for delivery, é prazo_entrega; se for salao, é tempo_espera_salao.
8. Em delivery, "chegou tudo certinho", "veio tudo direitinho" e frases parecidas sobre o pedido chegar em ordem são condicao_entrega positivo.
9. Demora ou rapidez da equipe para atender o cliente (por exemplo, "demoraram para me atender") é atendimento, não tempo_espera_salao.
10. atendimento é só a equipe presencial do salão. Simpatia, rapidez ou solução pelo WhatsApp, telefone ou app é resposta_canal."""


class Mention(BaseModel):
    aspect: Aspect
    polarity: Literal["positivo", "negativo"]
    evidencia: str = Field(
        description="Trecho copiado literalmente do texto da review que sustenta esta menção."
    )


class Extraction(BaseModel):
    mentions: list[Mention]


def build_system_prompt(version: str = PROMPT_VERSION) -> str:
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"Versão de prompt desconhecida: {version}")
    prompt = _base_prompt()
    if version == "extract_v2":
        prompt += "\n" + V2_RULES
    return prompt


def build_user_message(text: str, channel: str | None, version: str = PROMPT_VERSION) -> str:
    # O v1 não recebia o canal; mantemos assim para o v1 continuar reproduzível.
    if version == "extract_v1" or channel is None:
        return f"Review:\n{text}"
    return f"Canal: {channel}\nReview:\n{text}"


def _base_prompt() -> str:
    aspects = "\n".join(f"- {name}: {meaning}" for name, meaning in ASPECTS.items())
    return f"""Você analisa avaliações de clientes de restaurantes em português do Brasil e extrai, de forma estruturada, quais assuntos (aspectos) a review trata e com que polaridade.

ASSUNTOS (aspect) E O QUE SIGNIFICAM
{aspects}

REGRAS
1. Inclua uma menção SOMENTE para um assunto que o texto realmente sustenta. Não invente e não infira assunto que o texto não menciona.
2. "evidencia" tem que ser uma cópia LITERAL e contígua de um trecho da review (as mesmas palavras, na mesma ordem). Nunca parafraseie.
3. Cada aspecto aparece no máximo uma vez. Se o texto for ambíguo sobre um aspecto, escolha a polaridade que parece dominante.
4. Trate ironia pelo sentido real pretendido, não pelo sentido literal das palavras.
5. Satisfação ou insatisfação genérica, sem apontar um assunto específico (como "gostei", "foi ok", "não recomendo"), NÃO é uma menção a nenhum aspecto. Nesse caso, devolva uma lista vazia.
6. Se não houver nenhuma menção clara, devolva mentions como uma lista vazia. Uma lista vazia é uma resposta válida e esperada em boa parte dos casos."""


def evidence_is_verbatim(evidencia: str, text: str) -> bool:
    """A evidência tem que aparecer, palavra por palavra, dentro do texto original."""
    snippet = evidencia.strip()
    return bool(snippet) and snippet in text


def extract_one(
    client: OpenAI,
    model: str,
    text: str,
    channel: str | None = None,
    version: str = PROMPT_VERSION,
    effort: str = "low",
) -> dict:
    """Chama a API para uma review e devolve as menções já com o guard de evidência."""
    start = time.perf_counter()
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": build_system_prompt(version)},
            {"role": "user", "content": build_user_message(text, channel, version)},
        ],
        text_format=Extraction,
        reasoning={"effort": effort},
        max_output_tokens=1000,
    )
    latency = time.perf_counter() - start

    mentions = [
        {
            "aspect": m.aspect.value,
            "polarity": m.polarity,
            "evidencia": m.evidencia,
            "evidence_verified": evidence_is_verbatim(m.evidencia, text),
        }
        for m in response.output_parsed.mentions
    ]

    usage = response.usage
    return {
        "mentions": mentions,
        "model": model,
        "prompt_version": version,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost": call_cost(model, usage.input_tokens, usage.output_tokens),
        "latency_s": round(latency, 3),
    }
