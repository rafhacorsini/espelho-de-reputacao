"""Baseline de palavras-chave: o método mais simples que poderia resolver o problema.

Sem IA. Divide a review em trechos, procura palavras de cada aspecto e decide a
polaridade contando palavras boas e ruins no mesmo trecho. Escrito uma vez, sem
ajustar olhando o dev: se o LLM não vencer isto com folga, ele não vale o custo.
"""

import re
import unicodedata

ASPECT_KEYWORDS = {
    "atendimento": [
        "atendimento", "atendente", "garcom", "garconete", "equipe", "funcionario",
        "gerente", "atenderam", "atendeu", "atencios", "educad", "grosso", "grosseir",
    ],
    "tempo_espera_salao": [
        "fila", "espera", "esperamos", "esperei na mesa", "demorou pra sair",
        "demorou para sair", "chegar a mesa", "prato demorou", "sentar",
    ],
    "prazo_entrega": [
        "entrega", "entregue", "entregador", "motoboy", "motoqueiro", "pra chegar",
        "para chegar", "chegou rapid", "atrasou", "atraso", "prazo",
    ],
    "condicao_entrega": [
        "embalagem", "embalad", "frio", "fria", "vazou", "amassad", "faltou",
        "faltando", "veio errado", "pedido errado", "quentinh", "chegou quente",
        "temperatura",
    ],
    "qualidade_comida": [
        "comida", "sabor", "saboros", "gostos", "delicios", "tempero", "prato",
        "porcao", "pizza", "sushi", "hamburguer", "carne", "massa", "fresc",
        "insoss", "sem gosto", "cru", "passad", "risoto", "arroz", "lanche",
    ],
    "preco_valor": [
        "preco", "caro", "cara pelo", "barato", "custo-beneficio", "custo beneficio",
        "vale a pena", "valeu a pena", "valor", "salgado", "justo", "conta",
    ],
    "ambiente_limpeza": [
        "ambiente", "limp", "suj", "barulh", "aconcheg", "decoracao", "banheiro",
        "confortavel", "apertad", "clima do lugar",
    ],
    "resposta_canal": [
        "whatsapp", "zap", "telefone", "ligacao", "mensagem", "responder",
        "responderam", "resposta", "retorno", "contato", "pelo app",
    ],
}

POSITIVE = [
    "bom", "boa", "otim", "excelente", "maravilh", "perfeit", "rapid", "gostos",
    "delicios", "saboros", "fresc", "limp", "educad", "atencios", "prestativ",
    "caprich", "adorei", "amei", "recomendo", "justo", "barato", "vale a pena",
    "valeu a pena", "aconcheg", "agradavel", "confortavel", "certinh", "quentinh",
    "top", "show", "gostei", "curti",
]
NEGATIVE = [
    "ruim", "pessim", "horrivel", "demor", "lent", "frio", "fria", "caro", "salgad",
    "suj", "grosso", "grosseir", "mal ", "sem gosto", "insoss", "errado", "faltou",
    "faltando", "atras", "decepcion", "desagrad", "barulh", "apertad", "a desejar",
    "abaixo", "fraco", "seco", "seca", "nunca mais", "nao gostei", "chato",
]


def normalize(text: str) -> str:
    """Minúsculas e sem acento, para 'Pessimo' e 'péssimo' contarem igual."""
    without_accents = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return without_accents.lower()


def split_clauses(text: str) -> list[str]:
    return [c.strip() for c in re.split(r"[.!?;,]|\bmas\b|\bporem\b", text) if c.strip()]


def _count(clause: str, words: list[str]) -> int:
    return sum(clause.count(word) for word in words)


def clause_polarity(clause: str) -> int:
    """Positivo > 0, negativo < 0. 'não' antes de uma palavra boa inverte."""
    score = _count(clause, POSITIVE) - _count(clause, NEGATIVE)
    if re.search(r"\bnao\b\s+(\w+\s+)?(" + "|".join(POSITIVE) + ")", clause):
        score -= 2
    return score


def extract_baseline(text: str) -> list[dict]:
    clauses = [normalize(c) for c in split_clauses(text)]
    overall = sum(clause_polarity(c) for c in clauses)

    found = {}
    for clause in clauses:
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if aspect in found or not any(k in clause for k in keywords):
                continue
            score = clause_polarity(clause) or overall
            found[aspect] = "positivo" if score >= 0 else "negativo"
    return [{"aspect": a, "polarity": p} for a, p in found.items()]
