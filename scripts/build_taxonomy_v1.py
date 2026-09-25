"""Monta a taxonomia v1 com as decisões da curadoria humana (Dia 4, passos 6 e 7).

Decisões (25/09):
1. Base: os 8 aspectos × elogio/reclamação, vindos do extrator.
2. Subtema novo "prato em destaque" dentro de qualidade_comida. Os pratos foram
   DESCOBERTOS pelos grupos (risoto de camarão, hambúrguer); a DETECÇÃO é pelo
   nome do prato no trecho. Testamos detectar pelo centro do grupo, e ele perdia
   6 menções literais ao risoto e todas as reclamações do hambúrguer.
3. O grupo misturado "chegada do pedido" não é usado: quem separa prazo, espera
   e condição é o extrator (com o canal).

Uso:
    uv run python scripts/build_taxonomy_v1.py
"""

import json
from pathlib import Path

import pandas as pd

from espelho.schema import ASPECTS
from espelho.taxonomy import DISHES, detect_dish

SNIPPETS = Path("data/taxonomy/snippets.parquet")
OUT_SNIPPETS = Path("data/taxonomy/snippets_v1.parquet")
OUT_TAXONOMY = Path("data/taxonomy/taxonomy_v1.json")


def main() -> None:
    snippets = pd.read_parquet(SNIPPETS)
    is_food = snippets["aspect"] == "qualidade_comida"
    snippets["dish"] = [detect_dish(t) if food else None for t, food in zip(snippets["text"], is_food)]
    snippets["category"] = snippets["aspect"] + "|" + snippets["polarity"]
    snippets.to_parquet(OUT_SNIPPETS, index=False)

    print("Prato em destaque (trechos de comida):")
    for dish, (cluster_id, _) in DISHES.items():
        tagged = snippets[snippets["dish"] == dish]
        outside = (tagged["cluster"] != cluster_id).sum()
        by_polarity = tagged["polarity"].value_counts().to_dict()
        print(f"  {dish}: {len(tagged)} trechos {by_polarity} | {outside} fora do grupo original")

    taxonomy = {
        "version": "taxonomy_v1",
        "date": "2026-09-25",
        "decided_by": "curadoria humana sobre o rascunho de reports/taxonomy_draft.md",
        "categories": [
            {"aspect": a, "meaning": m, "polarities": ["positivo", "negativo"]} for a, m in ASPECTS.items()
        ],
        "subthemes": {
            "qualidade_comida": {
                "prato_em_destaque": {
                    dish: {"discovered_in_cluster": cid, "detected_by_words": words}
                    for dish, (cid, words) in DISHES.items()
                }
            }
        },
        "not_used": {"cluster 7 'chegada do pedido'": "mistura prazo, espera e condição; o extrator separa com o canal"},
    }
    OUT_TAXONOMY.write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT_TAXONOMY} e {OUT_SNIPPETS}")


if __name__ == "__main__":
    main()
