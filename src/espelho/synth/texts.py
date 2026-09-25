"""Lê e valida os textos colados de volta do chat.

Chats erram: linha quebrada, id trocado, texto repetido. Antes de usar qualquer
texto, validamos. Dado ruim aqui vira avaliação ruim lá na frente.
"""

import json
import re

MIN_LEN = 5
MAX_LEN = 900
# nomes técnicos como prazo_entrega não podem vazar para o texto
SNAKE_CASE = re.compile(r"\b[a-z]+_[a-z_]+\b")


def parse_lines(raw: str):
    """Lê a saída colada: uma linha JSON por review. Tolera cercas de código e vírgulas."""
    records, errors = [], []
    for number, line in enumerate(raw.splitlines(), start=1):
        line = line.strip().rstrip(",")
        if not line or line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            errors.append((number, "linha não é JSON válido"))
            continue
        if not isinstance(obj, dict) or "review_id" not in obj or "text" not in obj:
            errors.append((number, "faltam os campos review_id e text"))
            continue
        records.append(
            {"review_id": str(obj["review_id"]).strip(), "text": str(obj["text"]).strip()}
        )
    return records, errors


def validate(records, expected_ids, allow_duplicate_text: bool = False):
    """Separa os registros bons dos problemáticos.

    Devolve (válidos, problemas), onde cada problema é (review_id, motivo).
    No corpus inteiro, textos curtos repetidos ("Foi ok.") são realistas, por
    isso dá para permitir texto repetido com allow_duplicate_text.
    """
    expected = set(expected_ids)
    valid, problems, seen_ids, seen_texts = [], [], set(), set()
    for record in records:
        review_id, text = record["review_id"], record["text"]
        if review_id not in expected:
            problems.append((review_id, "id desconhecido"))
        elif review_id in seen_ids:
            problems.append((review_id, "id duplicado"))
        elif not (MIN_LEN <= len(text) <= MAX_LEN):
            problems.append((review_id, "tamanho fora do esperado"))
        elif SNAKE_CASE.search(text):
            problems.append((review_id, "vazou nome técnico de aspecto"))
        elif text.lower() in seen_texts and not allow_duplicate_text:
            problems.append((review_id, "texto repetido"))
        else:
            seen_ids.add(review_id)
            seen_texts.add(text.lower())
            valid.append(record)
    return valid, problems
