from espelho.extract import Aspect, evidence_is_verbatim
from espelho.schema import ASPECTS


def test_aspect_enum_matches_schema():
    assert {a.value for a in Aspect} == set(ASPECTS)


def test_evidence_verbatim_accepts_exact_snippet():
    text = "Chegou frio e o atendimento foi grosseiro."
    assert evidence_is_verbatim("atendimento foi grosseiro", text)


def test_evidence_verbatim_rejects_paraphrase():
    text = "Chegou frio e o atendimento foi grosseiro."
    assert not evidence_is_verbatim("o garçom foi mal educado", text)


def test_evidence_verbatim_rejects_empty():
    assert not evidence_is_verbatim("   ", "qualquer texto")
