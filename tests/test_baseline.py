from espelho.baseline import extract_baseline, normalize


def aspects(text):
    return {(m["aspect"], m["polarity"]) for m in extract_baseline(text)}


def test_normalize_removes_accents():
    assert normalize("Péssimo ATENDIMENTO") == "pessimo atendimento"


def test_simple_complaint():
    assert ("prazo_entrega", "negativo") in aspects("A entrega demorou demais.")


def test_mixed_review_splits_on_mas():
    result = aspects("A comida estava ótima, mas o preço é caro.")
    assert ("qualidade_comida", "positivo") in result
    assert ("preco_valor", "negativo") in result


def test_generic_review_has_no_aspect():
    assert aspects("Gostei, voltaria.") == set()
