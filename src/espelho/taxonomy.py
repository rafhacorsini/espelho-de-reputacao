"""Taxonomia descoberta dos dados (Dia 4).

Ideia em 4 passos:
1. Cada trecho de evidência vira um embedding: uma lista de números que
   representa o significado. Trechos parecidos ficam perto um do outro.
2. Reduzimos as dimensões (PCA) para o agrupamento funcionar melhor.
3. O HDBSCAN acha os grupos sozinho, pela densidade, e marca o que não cabe
   em nenhum grupo como ruído (-1).
4. Um LLM lê exemplos de cada grupo e dá nome e definição. Depois, você revisa.
"""

import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from espelho.cost import call_cost, embedding_cost
from espelho.schema import ASPECTS

EMBEDDING_MODEL = "text-embedding-3-small"
NAMING_MODEL = "gpt-5.6-terra"


def embed(client: OpenAI, texts: list[str], batch_size: int = 512) -> tuple[np.ndarray, float]:
    """Embeddings normalizados (tamanho 1), para a distância refletir o ângulo (cosseno)."""
    vectors, tokens = [], 0
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts[start : start + batch_size])
        vectors.extend(item.embedding for item in response.data)
        tokens += response.usage.total_tokens
    matrix = np.array(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix, embedding_cost(EMBEDDING_MODEL, tokens)


def reduce(matrix: np.ndarray, dims: int = 50, seed: int = 0) -> np.ndarray:
    return PCA(n_components=min(dims, matrix.shape[1]), random_state=seed).fit_transform(matrix)


def cluster(points: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Rótulo de grupo para cada ponto; -1 significa ruído (não coube em grupo nenhum)."""
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5).fit_predict(points)


def purity(clusters: np.ndarray, labels: list[str]) -> float:
    """Em cada grupo, a fração do rótulo mais comum; média ponderada pelo tamanho (sem ruído)."""
    kept = clusters >= 0
    if not kept.any():
        return 0.0
    total = 0
    for c in np.unique(clusters[kept]):
        members = [labels[i] for i in np.flatnonzero(clusters == c)]
        total += max(members.count(value) for value in set(members))
    return total / kept.sum()


def agreement_with_labels(clusters: np.ndarray, labels: list[str]) -> dict:
    kept = clusters >= 0
    return {
        "clusters": int(len(np.unique(clusters[kept]))),
        "noise_share": float(1 - kept.mean()),
        "purity": purity(clusters, labels),
        "adjusted_rand": float(adjusted_rand_score([labels[i] for i in np.flatnonzero(kept)], clusters[kept])),
    }


def representatives(points: np.ndarray, clusters: np.ndarray, cluster_id: int, texts: list[str], k: int = 15) -> list[str]:
    """Os k trechos mais próximos do centro do grupo, sem repetir texto."""
    idx = np.flatnonzero(clusters == cluster_id)
    center = points[idx].mean(axis=0)
    order = idx[np.argsort(np.linalg.norm(points[idx] - center, axis=1))]
    chosen, seen = [], set()
    for i in order:
        if texts[i].lower() not in seen:
            seen.add(texts[i].lower())
            chosen.append(texts[i])
        if len(chosen) == k:
            break
    return chosen


class ClusterName(BaseModel):
    name: str = Field(description="Nome curto da dor ou força, em português, de 2 a 5 palavras.")
    definition: str = Field(description="Uma frase explicando o que os trechos têm em comum.")
    polarity: str = Field(description="'positivo', 'negativo' ou 'misto'.")
    parent_aspect: str = Field(description=f"O aspecto mais próximo, um de: {', '.join(ASPECTS)}.")


def name_cluster(client: OpenAI, snippets: list[str]) -> tuple[dict, float]:
    examples = "\n".join(f"- {s}" for s in snippets)
    response = client.responses.parse(
        model=NAMING_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "Você recebe trechos de avaliações de restaurantes que um algoritmo agrupou por "
                    "significado. Diga o que eles têm em comum, de forma específica (por exemplo, "
                    "'demora na resposta do WhatsApp' em vez de 'demora'). Aspectos possíveis: "
                    + ", ".join(ASPECTS)
                ),
            },
            {"role": "user", "content": f"Trechos do grupo:\n{examples}"},
        ],
        text_format=ClusterName,
        reasoning={"effort": "low"},
        max_output_tokens=800,
    )
    usage = response.usage
    return response.output_parsed.model_dump(), call_cost(NAMING_MODEL, usage.input_tokens, usage.output_tokens)
