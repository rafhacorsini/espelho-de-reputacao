"""Destilação e cascata (Dia 6, parte A).

Destilação: o LLM (professor) já rotulou as reviews; um modelo pequeno (aluno)
aprende a imitar esses rótulos a partir do embedding da review. O aluno é uma
regressão logística por aspecto, com 3 classes: nenhum, positivo, negativo.

Cascata: o aluno responde primeiro. Se ele estiver inseguro em algum aspecto
(probabilidade máxima abaixo de um limiar), a review sobe para o LLM.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from espelho.schema import ASPECTS

CLASSES = ["nenhum", "positivo", "negativo"]


def mentions_to_classes(mentions) -> dict[str, str]:
    """Para cada aspecto, a classe que a review tem."""
    found = {m["aspect"]: m["polarity"] for m in mentions}
    return {aspect: found.get(aspect, "nenhum") for aspect in ASPECTS}


class _Constant:
    """Modelo para um aspecto com uma classe só no treino: sempre responde ela, com certeza."""

    def __init__(self, cls: str):
        self.classes_ = np.array([cls])

    def predict_proba(self, X):
        return np.ones((len(X), 1))


def train_student(X: np.ndarray, teacher_mentions: list) -> dict:
    """Um modelo por aspecto."""
    labels = [mentions_to_classes(m) for m in teacher_mentions]
    models = {}
    for aspect in ASPECTS:
        y = [row[aspect] for row in labels]
        if len(set(y)) == 1:
            models[aspect] = _Constant(y[0])
        else:
            models[aspect] = LogisticRegression(max_iter=2000).fit(X, y)
    return models


def student_predict(models: dict, X: np.ndarray) -> tuple[list[list[dict]], np.ndarray]:
    """Menções previstas e a confiança de cada review (a do aspecto mais incerto)."""
    mentions = [[] for _ in range(len(X))]
    confidence = np.ones(len(X))
    for aspect, model in models.items():
        probs = model.predict_proba(X)
        best = probs.argmax(axis=1)
        confidence = np.minimum(confidence, probs.max(axis=1))
        for i, cls_index in enumerate(best):
            cls = model.classes_[cls_index]
            if cls != "nenhum":
                mentions[i].append({"aspect": aspect, "polarity": cls})
    return mentions, confidence


def cascade(student: list, llm: list, confidence: np.ndarray, tau: float) -> tuple[list, float]:
    """Usa o aluno quando a confiança é >= tau; senão, o LLM. Devolve (menções, fração que foi ao LLM)."""
    to_llm = confidence < tau
    merged = [llm[i] if to_llm[i] else student[i] for i in range(len(student))]
    return merged, float(to_llm.mean())
