"""Descobre a taxonomia de dores e forças a partir dos trechos de evidência (Dia 4).

Uso:
    uv run python scripts/discover_taxonomy.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from espelho.taxonomy import (
    agreement_with_labels,
    cluster,
    embed,
    name_cluster,
    reduce,
    representatives,
)

PREDICTIONS = Path("data/predictions/corpus_extract_v2.parquet")
OUT_DIR = Path("data/taxonomy")
EMBEDDINGS = OUT_DIR / "embeddings.npy"
SNIPPETS = OUT_DIR / "snippets.parquet"
REPORT = Path("reports/taxonomy_draft.md")

# Regra de escolha decidida antes de ver os resultados: o tamanho mínimo de
# grupo com maior Rand ajustado contra os rótulos do extrator.
MIN_CLUSTER_SIZES = [15, 30, 60]
DELAY_LABELS = ["prazo_entrega|negativo", "tempo_espera_salao|negativo", "resposta_canal|negativo"]


def load_snippets() -> pd.DataFrame:
    preds = pd.read_parquet(PREDICTIONS)
    rows = [
        {"review_id": r.review_id, "aspect": m["aspect"], "polarity": m["polarity"], "text": m["evidencia"]}
        for r in preds.itertuples()
        for m in r.pred_mentions
        if m["evidence_verified"]
    ]
    df = pd.DataFrame(rows)
    df["label"] = df["aspect"] + "|" + df["polarity"]
    return df


def main() -> None:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY"), max_retries=5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    snippets = load_snippets()
    print(f"{len(snippets)} trechos de evidência ({snippets['text'].nunique()} textos diferentes)")

    total_cost = 0.0
    if EMBEDDINGS.exists() and SNIPPETS.exists() and len(pd.read_parquet(SNIPPETS)) == len(snippets):
        matrix = np.load(EMBEDDINGS)
        print("Embeddings reaproveitados do disco (custo zero)")
    else:
        matrix, cost = embed(client, snippets["text"].tolist())
        np.save(EMBEDDINGS, matrix)
        total_cost += cost
        print(f"Embeddings: {matrix.shape[0]} vetores de {matrix.shape[1]} números | custo US$ {cost:.5f}")

    points = reduce(matrix)
    labels = snippets["label"].tolist()

    print("\nTamanho mínimo de grupo -> resultado (comparado com os rótulos aspecto|polaridade):")
    results = {}
    for size in MIN_CLUSTER_SIZES:
        clusters = cluster(points, size)
        results[size] = (clusters, agreement_with_labels(clusters, labels))
        a = results[size][1]
        print(
            f"  {size:>3}: {a['clusters']:>3} grupos | ruído {a['noise_share']:.0%} | "
            f"pureza {a['purity']:.2f} | Rand ajustado {a['adjusted_rand']:.2f}"
        )
    best = max(results, key=lambda s: results[s][1]["adjusted_rand"])
    clusters = results[best][0]
    print(f"Escolhido: {best}")

    snippets["cluster"] = clusters
    snippets["x"], snippets["y"] = reduce(matrix, dims=2)[:, 0], reduce(matrix, dims=2)[:, 1]

    delay = snippets[snippets["label"].isin(DELAY_LABELS) & (snippets["cluster"] >= 0)]
    print("\nA pergunta do dia: as demoras se separam? (linhas: rótulo; colunas: grupo)")
    print(pd.crosstab(delay["label"], delay["cluster"]).to_string())

    texts = snippets["text"].tolist()
    draft = []
    for cluster_id in sorted(c for c in np.unique(clusters) if c >= 0):
        reps = representatives(points, clusters, cluster_id, texts)
        naming, cost = name_cluster(client, reps)
        total_cost += cost
        members = snippets[snippets["cluster"] == cluster_id]
        draft.append(
            {
                "cluster": int(cluster_id),
                "size": int(len(members)),
                **naming,
                "top_label": members["label"].value_counts().index[0],
                "top_label_share": float(members["label"].value_counts(normalize=True).iloc[0]),
                "examples": reps[:3],
            }
        )

    snippets.to_parquet(SNIPPETS, index=False)
    (OUT_DIR / "taxonomy_draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Rascunho da taxonomia (Dia 4)",
        "",
        f"{len(snippets)} trechos, {len(draft)} grupos, ruído {results[best][1]['noise_share']:.0%}.",
        "",
        "| Grupo | Trechos | Nome (sugerido pelo LLM) | Polaridade | Aspecto pai | Exemplo |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for d in sorted(draft, key=lambda d: -d["size"]):
        lines.append(
            f"| {d['cluster']} | {d['size']} | {d['name']} | {d['polarity']} | {d['parent_aspect']} | {d['examples'][0]} |"
        )
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n{len(draft)} grupos nomeados -> {REPORT}")
    print(f"Custo total do Dia 4 (embeddings + nomes): US$ {total_cost:.4f}")


if __name__ == "__main__":
    main()
