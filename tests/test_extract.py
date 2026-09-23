import pytest

from espelho.extract import Aspect, build_system_prompt, build_user_message, evidence_is_verbatim
from espelho.schema import ASPECTS


def test_v1_prompt_has_no_v2_rules():
    assert "7. A mensagem informa o canal" not in build_system_prompt("extract_v1")
    assert "7. A mensagem informa o canal" in build_system_prompt("extract_v2")


def test_channel_only_goes_to_v2():
    assert "Canal" not in build_user_message("Demorou.", "delivery", "extract_v1")
    assert build_user_message("Demorou.", "delivery", "extract_v2").startswith("Canal: delivery")


def test_unknown_prompt_version_fails():
    with pytest.raises(ValueError):
        build_system_prompt("extract_v9")


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
