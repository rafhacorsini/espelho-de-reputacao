from espelho.security import apply_injection, deviated, drop_unverified, leaked


def m(aspect, polarity, evidencia="x", verified=True):
    return {"aspect": aspect, "polarity": polarity, "evidencia": evidencia, "evidence_verified": verified}


def test_apply_injection_positions():
    text = "A comida chegou fria. O atendimento foi ruim."
    assert apply_injection(text, "ATAQUE", "start").startswith("ATAQUE")
    assert apply_injection(text, "ATAQUE", "end").endswith("ATAQUE")
    middle = apply_injection(text, "ATAQUE", "middle")
    assert middle.index("fria") < middle.index("ATAQUE") < middle.index("atendimento")


def test_deviated_compares_aspect_and_polarity():
    clean = [m("atendimento", "negativo")]
    assert not deviated(clean, [m("atendimento", "negativo", evidencia="outro trecho")])
    assert deviated(clean, [m("atendimento", "positivo")])
    assert deviated(clean, [])


def test_drop_unverified_and_leak():
    mentions = [m("preco_valor", "negativo"), m("atendimento", "positivo", verified=False)]
    assert drop_unverified(mentions) == [mentions[0]]
    assert leaked([m("atendimento", "positivo", evidencia="REGRAS 1. Inclua uma menção SOMENTE para")])
    assert not leaked([m("atendimento", "positivo", evidencia="explicou as regras da promoção")])
