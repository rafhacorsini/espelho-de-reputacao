import pandas as pd
import pytest

from espelho.synth.prompts import build_prompt, select_pilot
from espelho.synth.texts import parse_lines, validate
from espelho.synth.truth import generate_truth


@pytest.fixture(scope="module")
def truth():
    return generate_truth(seed=42)


def test_parse_lines_tolerates_code_fences_and_trailing_commas():
    raw = '```json\n{"review_id": "R1", "text": "Gostei muito."},\n\n{"review_id": "R2", "text": "Ok."}\n```'
    records, errors = parse_lines(raw)
    assert [r["review_id"] for r in records] == ["R1", "R2"]
    assert errors == []


def test_parse_lines_reports_broken_lines():
    records, errors = parse_lines('{"review_id": "R1", "text": "ok"}\nisso não é json\n{"review_id": "R3"}')
    assert len(records) == 1
    assert [number for number, _ in errors] == [2, 3]


def test_validate_flags_bad_records():
    records = [
        {"review_id": "R1", "text": "Comida boa demais, voltarei."},
        {"review_id": "R1", "text": "Outro texto para o mesmo id."},
        {"review_id": "R9", "text": "Id que não existe no lote."},
        {"review_id": "R2", "text": "Ok"},
        {"review_id": "R3", "text": "O prazo_entrega foi ruim."},
        {"review_id": "R4", "text": "Comida boa demais, voltarei."},
    ]
    valid, problems = validate(records, ["R1", "R2", "R3", "R4"])
    assert [r["review_id"] for r in valid] == ["R1"]
    reasons = dict(problems)
    assert reasons["R9"] == "id desconhecido"
    assert reasons["R2"] == "tamanho fora do esperado"
    assert reasons["R3"] == "vazou nome técnico de aspecto"
    assert reasons["R4"] == "texto repetido"


def test_pilot_has_the_hard_cases(truth):
    pilot = select_pilot(truth, n=180)
    assert len(pilot) == 180
    assert (pilot["n_mentions"] == 0).sum() >= 20
    assert (pilot["sarcastic"] & (pilot["n_mentions"] > 0)).sum() >= 20
    assert pilot["store_id"].nunique() == 6


def test_prompt_contains_every_review_and_every_aspect(truth):
    rows = truth.head(5)
    prompt = build_prompt(rows)
    for review_id in rows["review_id"]:
        assert review_id in prompt
    assert "prazo_entrega" in prompt
    assert "{rows}" not in prompt and "{aspects}" not in prompt
