"""Ferramenta local para você rotular as reviews à mão (Dia 1, passo 6).

Rodar com:
    uv run streamlit run app/label_reviews.py

Cada review mostra o texto, e para cada um dos 8 aspectos você marca:
não mencionado, elogio ou reclamação. Salva a cada clique em "Salvar e avançar".

Rodada cega (Dia 3): rotula de novo as 50 reviews do test, sem ver os rótulos
antigos e já com as regras do gabarito v2:
    uv run streamlit run app/label_reviews.py -- --rodada-cega
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from espelho.gold import TARGET_LABELED, load_labels, progress, save_label  # noqa: E402
from espelho.schema import ASPECTS  # noqa: E402

BLIND = "--rodada-cega" in sys.argv

if BLIND:
    PILOT_REVIEWS = Path("data/gold/test.parquet")
    LABELS_PATH = Path("data/gold/relabel_test.jsonl")
    TARGET = 50
else:
    PILOT_REVIEWS = Path("data/synthetic/pilot_reviews.parquet")
    LABELS_PATH = Path("data/gold/labels.jsonl")
    TARGET = TARGET_LABELED

RULES_V2 = """**Regras do gabarito v2** (além das de sempre):
- Rapidez ou demora sem dizer onde: **delivery → prazo de entrega**, **salão → tempo de espera no salão**.
- Em delivery, "chegou tudo certinho" / "veio tudo direitinho" → **condição da entrega, elogio**.
- Demora ou rapidez da equipe para atender → **atendimento**.
- Atendimento é só a equipe do salão. Simpatia ou solução pelo WhatsApp, telefone ou app → **resposta do canal**.
- Satisfação genérica, sem dizer o quê → nenhum aspecto."""

OPTIONS = ["não mencionado", "elogio", "reclamação"]
POLARITY_BY_OPTION = {"elogio": "positivo", "reclamação": "negativo"}


@st.cache_data
def load_pilot(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def main() -> None:
    st.set_page_config(page_title="Rotular reviews", layout="centered")

    if not PILOT_REVIEWS.exists():
        st.error(f"Não encontrei {PILOT_REVIEWS}. Rode a importação dos textos primeiro.")
        return

    df = load_pilot(str(PILOT_REVIEWS))
    labels = load_labels(LABELS_PATH)
    stats = progress(labels, target=TARGET)

    st.title("Rodada cega: test" if BLIND else "Rotular reviews")
    if BLIND:
        st.info(RULES_V2)
    st.progress(min(1.0, stats["labeled"] / stats["target"]))
    st.caption(
        f"{stats['labeled']} de {stats['target']} rotuladas "
        f"(mais {stats['discarded']} descartadas). Faltam {stats['remaining']}."
    )
    if stats["done"]:
        st.success("Meta atingida! Você pode continuar rotulando as reviews que sobraram, se quiser.")

    if "index" not in st.session_state:
        unlabeled = df[~df["review_id"].isin(labels)]
        st.session_state.index = int(unlabeled.index[0]) if len(unlabeled) else 0

    st.session_state.index = st.number_input(
        "Review nº (posição na lista)", min_value=0, max_value=len(df) - 1,
        value=st.session_state.index, step=1,
    )
    row = df.iloc[st.session_state.index]
    existing = labels.get(row["review_id"])

    st.divider()
    cuisine = f" · {row['cuisine']}" if "cuisine" in row else ""
    st.markdown(f"**{row['review_id']}** · {row['channel']}{cuisine} · {row['stars']}★")
    st.markdown(f"> {row['text']}")
    if existing:
        st.info("Esta review já foi rotulada. Você pode ver e corrigir os valores abaixo.")

    existing_by_aspect = {}
    if existing and not existing.get("discarded"):
        existing_by_aspect = {m["aspect"]: m["polarity"] for m in existing["mentions"]}
    reverse_option = {"positivo": "elogio", "negativo": "reclamação"}

    choices = {}
    for aspect, meaning in ASPECTS.items():
        default = reverse_option.get(existing_by_aspect.get(aspect), "não mencionado")
        choices[aspect] = st.radio(
            f"**{aspect.replace('_', ' ')}** — {meaning}",
            OPTIONS, index=OPTIONS.index(default), horizontal=True,
            key=f"aspect_{row['review_id']}_{aspect}",
        )

    col1, col2, col3 = st.columns(3)
    if col1.button("Salvar e avançar", type="primary"):
        mentions = [
            {"aspect": aspect, "polarity": POLARITY_BY_OPTION[choice]}
            for aspect, choice in choices.items() if choice != "não mencionado"
        ]
        save_label(LABELS_PATH, row["review_id"], mentions)
        st.session_state.index = min(st.session_state.index + 1, len(df) - 1)
        st.rerun()

    if col2.button("Descartar (texto ruim)"):
        save_label(LABELS_PATH, row["review_id"], [], discarded=True)
        st.session_state.index = min(st.session_state.index + 1, len(df) - 1)
        st.rerun()

    if col3.button("Pular sem salvar"):
        st.session_state.index = min(st.session_state.index + 1, len(df) - 1)
        st.rerun()


if __name__ == "__main__":
    main()
