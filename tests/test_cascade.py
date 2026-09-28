import numpy as np

from espelho.cascade import cascade, mentions_to_classes, student_predict, train_student


def test_mentions_to_classes_fills_every_aspect():
    classes = mentions_to_classes([{"aspect": "preco_valor", "polarity": "negativo"}])
    assert classes["preco_valor"] == "negativo"
    assert classes["atendimento"] == "nenhum"
    assert len(classes) == 8


def test_student_learns_a_separable_aspect():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    teacher = [[{"aspect": "preco_valor", "polarity": "negativo"}] if x[0] > 0 else [] for x in X]
    models = train_student(X, teacher)
    preds, confidence = student_predict(models, np.array([[3, 0, 0, 0], [-3, 0, 0, 0]]))
    assert preds[0] == [{"aspect": "preco_valor", "polarity": "negativo"}]
    assert preds[1] == []
    assert confidence.min() > 0.8


def test_cascade_sends_unsure_reviews_to_llm():
    student = [["aluno"], ["aluno"], ["aluno"]]
    llm = [["llm"], ["llm"], ["llm"]]
    merged, share = cascade(student, llm, np.array([0.99, 0.60, 0.95]), tau=0.9)
    assert merged == [["aluno"], ["llm"], ["aluno"]]
    assert share == 1 / 3
