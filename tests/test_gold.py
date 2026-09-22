from espelho.gold import load_labels, progress, save_label


def test_round_trip(tmp_path):
    path = tmp_path / "labels.jsonl"
    save_label(path, "R1", [{"aspect": "qualidade_comida", "polarity": "positivo"}])
    save_label(path, "R2", [], discarded=True)

    labels = load_labels(path)
    assert labels["R1"]["mentions"][0]["aspect"] == "qualidade_comida"
    assert labels["R2"]["discarded"] is True


def test_last_write_wins(tmp_path):
    path = tmp_path / "labels.jsonl"
    save_label(path, "R1", [{"aspect": "atendimento", "polarity": "negativo"}])
    save_label(path, "R1", [{"aspect": "atendimento", "polarity": "positivo"}])

    labels = load_labels(path)
    assert labels["R1"]["mentions"][0]["polarity"] == "positivo"


def test_progress_counts_labeled_and_discarded():
    labels = {
        "R1": {"mentions": [], "discarded": False},
        "R2": {"mentions": [], "discarded": False},
        "R3": {"mentions": [], "discarded": True},
    }
    result = progress(labels, target=2)
    assert result == {"labeled": 2, "discarded": 1, "target": 2, "remaining": 0, "done": True}
