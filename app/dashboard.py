"""Dashboard público do case Espelho de Reputação.

Só lê arquivos pré-calculados e versionados: nenhuma chave de API é necessária.

Rodar com:
    uv run streamlit run app/dashboard.py
"""

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from espelho.analytics import binomial_cusum, indicator_table, mean_cusum_down  # noqa: E402
from espelho.synth.truth import EVENTS  # noqa: E402

ASPECT_NAMES = {
    "atendimento": "Atendimento", "tempo_espera_salao": "Espera no salão", "prazo_entrega": "Prazo de entrega",
    "condicao_entrega": "Condição da entrega", "qualidade_comida": "Comida", "preco_valor": "Preço",
    "ambiente_limpeza": "Ambiente", "resposta_canal": "Resposta no WhatsApp/app",
}
POLARITY_NAMES = {"positivo": "elogio", "negativo": "reclamação"}

# Paleta de referência do guia de visualização, validada nos dois modos.
THEMES = {
    "light": {"series": "#2a78d6", "neg": "#e34948", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"series": "#3987e5", "neg": "#e66767", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835"},
}
BASELINE_DAYS, H, WINDOW = 15, 5.0, 14


def colors() -> dict:
    kind = getattr(getattr(st.context, "theme", None), "type", None)
    return THEMES["dark" if kind == "dark" else "light"]


def style(chart: alt.Chart, c: dict) -> alt.Chart:
    return (
        chart.configure_view(stroke=None)
        .configure_axis(gridColor=c["grid"], domainColor=c["axis"], tickColor=c["axis"], labelColor=c["muted"],
                        titleColor=c["ink2"], labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
        .configure_legend(labelColor=c["ink2"], titleColor=c["ink2"], orient="top")
    )


@st.cache_data
def load_summary() -> dict:
    return json.loads((ROOT / "reports/dashboard_summary.json").read_text(encoding="utf-8"))


@st.cache_data
def load_corpus() -> tuple[pd.DataFrame, pd.DataFrame]:
    corpus = pd.read_parquet(ROOT / "data/synthetic/corpus_reviews.parquet")
    preds = pd.read_parquet(ROOT / "data/predictions/corpus_extract_v2.parquet")[["review_id", "pred_mentions"]]
    corpus = corpus.merge(preds, on="review_id").reset_index(drop=True)
    return corpus, indicator_table(corpus["pred_mentions"])


def label(category: str) -> str:
    aspect, polarity = category.split("|")
    return f"{ASPECT_NAMES[aspect]} · {POLARITY_NAMES[polarity]}"


@st.cache_data
def event_timeline(event_id: str) -> tuple[pd.DataFrame, int, int | None, int | None]:
    corpus, indicators = load_corpus()
    event = next(e for e in EVENTS if e.event_id == event_id)
    rows = corpus if event.store_id is None else corpus[corpus["store_id"] == event.store_id]
    category = f"{event.aspect}|{event.polarity}"
    days = np.arange(corpus["day"].max() + 1)
    totals = rows.groupby("day").size().reindex(days, fill_value=0)
    counts = indicators.loc[rows.index, category].groupby(rows["day"]).sum().reindex(days, fill_value=0)
    star_sum = rows.groupby("day")["stars"].sum().reindex(days, fill_value=0)

    aspect_alarms = binomial_cusum(counts.to_numpy(), totals.to_numpy(), BASELINE_DAYS, H)
    star_values = [rows.loc[rows["day"] == d, "stars"].to_numpy(dtype=float) for d in days]
    star_alarms = mean_cusum_down(star_values, BASELINE_DAYS, h=H)
    aspect_alarm = next((a for a in aspect_alarms if event.start_day <= a <= event.start_day + WINDOW), None)
    star_alarm = next((a for a in star_alarms if a >= event.start_day), None)

    df = pd.DataFrame({
        "dia": days,
        "taxa": counts.rolling(7, min_periods=1).sum() / totals.rolling(7, min_periods=1).sum(),
        "nota": star_sum.rolling(7, min_periods=1).sum() / totals.rolling(7, min_periods=1).sum(),
    })
    return df, event.start_day, aspect_alarm, star_alarm


def markers(start: int, aspect_alarm, star_alarm) -> pd.DataFrame:
    rows = [{"dia": start, "o_que": "início do problema", "tipo": "início"}]
    if aspect_alarm is not None:
        rows.append({"dia": aspect_alarm, "o_que": f"alarme por aspecto (+{aspect_alarm - start} dias)", "tipo": "aspecto"})
    if star_alarm is not None:
        rows.append({"dia": star_alarm, "o_que": f"alarme pela nota média (+{star_alarm - start} dias)", "tipo": "nota"})
    return pd.DataFrame(rows)


def timeline_chart(df: pd.DataFrame, marks: pd.DataFrame, y: str, title: str, fmt: str, c: dict, zero: bool) -> alt.Chart:
    x = alt.X("dia:Q", title="dia", scale=alt.Scale(domain=[0, int(df["dia"].max())]))
    # A nota média é um nível: começar no zero achataria justamente a queda que queremos mostrar.
    y_enc = alt.Y(f"{y}:Q", title=title, axis=alt.Axis(format=fmt, tickCount=4), scale=alt.Scale(zero=zero))
    line = alt.Chart(df).mark_line(strokeWidth=2, color=c["series"]).encode(x=x, y=y_enc)
    hover = alt.Chart(df).mark_point(size=250, opacity=0).encode(
        x=x, y=f"{y}:Q", tooltip=[alt.Tooltip("dia:Q", title="dia"), alt.Tooltip(f"{y}:Q", title=title, format=fmt)]
    )
    dash = alt.condition(alt.datum.tipo == "início", alt.value([4, 4]), alt.value([1, 0]))
    ink = alt.condition(alt.datum.tipo == "nota", alt.value(c["muted"]), alt.value(c["ink2"]))
    rules = alt.Chart(marks).mark_rule(strokeWidth=1.5).encode(x=x, strokeDash=dash, color=ink, tooltip=["o_que:N"])
    return (rules + line + hover).properties(height=190)


def rule_labels(marks: pd.DataFrame, c: dict) -> str:
    return " · ".join(f"**dia {int(r.dia)}**: {r.o_que}" for r in marks.itertuples())


def impact_chart(impact: list[dict], c: dict) -> alt.Chart:
    df = pd.DataFrame(impact)
    df["categoria"] = df["category"].map(label)
    df["tipo"] = np.where(df["pipeline"] < 0, "reclamação", "elogio")
    order = df.sort_values("pipeline")["categoria"].tolist()
    # Todos os 16 rótulos visíveis e inteiros: sem isso o gráfico esconde metade e corta os nomes.
    y = alt.Y("categoria:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False))
    color = alt.Color("tipo:N", scale=alt.Scale(domain=["elogio", "reclamação"], range=[c["series"], c["neg"]]), title=None)
    tooltip = [alt.Tooltip("categoria:N", title="menção"), alt.Tooltip("pipeline:Q", title="estrelas", format="+.2f"),
               alt.Tooltip("pipeline_low:Q", title="mín. (95%)", format="+.2f"), alt.Tooltip("pipeline_high:Q", title="máx. (95%)", format="+.2f"),
               alt.Tooltip("plantado:Q", title="plantado", format="+.2f")]
    bars = alt.Chart(df).mark_bar(height=12, cornerRadius=2).encode(
        x=alt.X("pipeline:Q", title="estrelas a mais ou a menos na nota", axis=alt.Axis(format="+.1f")), y=y, color=color, tooltip=tooltip)
    ci = alt.Chart(df).mark_rule(strokeWidth=1.5, color=c["ink2"]).encode(x="pipeline_low:Q", x2="pipeline_high:Q", y=y)
    planted = alt.Chart(df).mark_tick(thickness=2, size=16, color=c["ink"]).encode(x="plantado:Q", y=y, tooltip=tooltip)
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=c["axis"]).encode(x="x:Q")
    return (zero + bars + ci + planted).properties(height=16 * 30)


def owner_sections() -> dict:
    text = (ROOT / "reports/owner_report.md").read_text(encoding="utf-8")
    parts = text.split("\n## ")[1:]
    return {p.split(" (")[0]: "### " + p for p in parts}


def main() -> None:
    st.set_page_config(page_title="Espelho de Reputação", layout="wide")
    c = colors()
    s = load_summary()
    official = s["evaluation_test"][0]["f1"]
    baseline = next(e["f1"] for e in s["evaluation_test"] if e["system"].startswith("Baseline"))

    st.title("Espelho de Reputação")
    st.caption("Case de Engenharia de IA: transforma reviews de restaurantes em dores e forças mensuráveis, "
               "avisa quando um problema começa e mostra quanto cada dor custa em estrelas. Dados sintéticos com "
               "verdade plantada; validação com reviews reais em andamento.")

    tabs = st.tabs(["Visão geral", "Alarme antecipado", "Impacto em estrelas", "Relatório por loja", "Qualidade e custo", "Segurança"])

    with tabs[0]:
        a, b, d, e = st.columns(4)
        a.metric("F1 no test cego", f"{official[0]:.2f}", f"{official[0] - baseline[0]:+.2f} vs. baseline", delta_color="off")
        b.metric("Aviso antes da nota média", "2 a 3 semanas",
                 help="Nos 3 problemas de uma loja só: alarme em 2 a 8 dias contra 16 a 27 da nota média. "
                      "No aumento de preço na rede toda, a diferença foi de 2 dias.")
        d.metric("Concordância humana (κ)", f"{s['kappa']:.2f}")
        e.metric("Custo total de API do projeto", f"US$ {sum(x['usd'] for x in s['costs']):.2f}")
        st.markdown(
            "**Como funciona:** 3.555 reviews → um LLM extrai *aspecto, elogio/reclamação e o trecho literal* "
            "(o código confere se o trecho existe no texto) → estatística mede o impacto em estrelas e dispara um "
            "alarme quando uma reclamação cresce → embeddings descobrem temas novos (achou sozinho o risoto de "
            "camarão plantado na Loja 2).\n\n"
            "**Por que os números valem:** o gabarito foi rotulado à mão, o test foi aberto uma única vez com o "
            "gabarito congelado no git antes, e cada número tem intervalo de confiança. Os erros e as limitações "
            "estão documentados no README."
        )
        st.dataframe(pd.DataFrame(s["costs"]).rename(columns={"day": "dia", "what": "o quê", "usd": "US$"}),
                     hide_index=True, width="stretch")

    with tabs[1]:
        st.markdown("Cinco problemas foram plantados nos dados. O alarme por aspecto vigia cada tipo de reclamação "
                    "por loja; o jeito comum vigia só a nota média.")
        options = {f"{e.event_id} · {e.description}": e.event_id for e in EVENTS}
        choice = st.selectbox("Evento plantado", list(options), index=0)
        df, start, aspect_alarm, star_alarm = event_timeline(options[choice])
        marks = markers(start, aspect_alarm, star_alarm)
        event = next(e for e in EVENTS if e.event_id == options[choice])
        st.markdown(rule_labels(marks, c))
        st.markdown(f"**Reviews com \"{label(event.aspect + '|' + event.polarity)}\"** (média de 7 dias)")
        st.altair_chart(style(timeline_chart(df, marks, "taxa", "% das reviews", ".0%", c, zero=True), c), width="stretch")
        st.markdown("**Nota média da loja** (média de 7 dias)")
        st.altair_chart(style(timeline_chart(df, marks, "nota", "estrelas", ".1f", c, zero=False), c), width="stretch")
        val = s["alarm"]["validation"]
        st.caption(f"Validado num mundo novo, que o alarme nunca viu: achou {len(val['detections'])} de 5 eventos, "
                   f"com {val['false_per_1000_series_days']:.1f} alarmes falsos por 1.000 séries-dia.")
        with st.expander("Ver como tabela"):
            st.dataframe(df.round(3), hide_index=True, width="stretch")

    with tabs[2]:
        st.markdown("Quanto cada menção muda a nota, em média, mantendo o resto igual (regressão linear). "
                    "A barra é o que o sistema mediu, a linha fina é o intervalo de 95% e o traço é o efeito que "
                    "plantamos. É associação, não causa.")
        st.altair_chart(style(impact_chart(s["impact"], c), c), width="stretch")
        with st.expander("Ver como tabela"):
            st.dataframe(pd.DataFrame(s["impact"]).round(2), hide_index=True, width="stretch")

    with tabs[3]:
        sections = owner_sections()
        store = st.selectbox("Loja", list(sections))
        st.markdown(sections[store])
        st.caption("Últimos 14 dias. Os pratos em destaque vêm dos temas descobertos por embeddings.")

    with tabs[4]:
        st.markdown(f"**Avaliação no test cego** ({s['test_reviews']} reviews, gabarito com dupla rotulagem):")
        st.dataframe(pd.DataFrame([{"sistema": e["system"], "F1": round(e["f1"][0], 2),
                                    "intervalo 95%": f"{e['f1'][1]:.2f} a {e['f1'][2]:.2f}"} for e in s["evaluation_test"]]),
                     hide_index=True, width="stretch")
        st.markdown("**Cascata de custo:** um modelo pequeno (aluno) aprende com o LLM e só manda ao LLM as reviews em que está inseguro.")
        st.dataframe(pd.DataFrame([{"sistema": r["system"], "F1 no test": round(r["f1_test"], 2),
                                    "US$ por 1.000 reviews": round(r["cost_per_1000"], 3),
                                    "vai ao LLM": f"{r['share_llm']:.0%}"} for r in s["cascade"]]),
                     hide_index=True, width="stretch")
        st.markdown("**Temas descobertos por embeddings** (HDBSCAN + nomes sugeridos por LLM, revisados à mão):")
        st.dataframe(pd.DataFrame(s["taxonomy"]).sort_values("size", ascending=False), hide_index=True, width="stretch")

    with tabs[5]:
        sec = s["security"]
        st.markdown("Reviews são texto de terceiros que entra no prompt. Atacamos o próprio sistema com 40 reviews "
                    "de injeção de prompt (10 estilos) e medimos cada defesa. Ruído de base (mesma review, duas "
                    f"rodadas, sem ataque): {sec['noise']:.0%}.")
        st.dataframe(pd.DataFrame([{"configuração": k, "ataques que mudaram a saída": f"{v['asr']:.0%}",
                                    "custo extra por review (US$)": f"{v['extra_cost']:.6f}"} for k, v in sec["configs"].items()]),
                     hide_index=True, width="stretch")
        g = sec["guard"]
        st.markdown(f"O guarda marcou {g['attacks_flagged']:.0%} dos ataques e bloqueou por engano "
                    f"{g['normal_false_blocks']} de {g['normal_total']} reviews normais. **Leitura honesta:** ataques "
                    "ingênuos quase não funcionam em modelos de 2026; ataques adaptativos não foram testados. A "
                    "defesa principal é a contenção: o extrator não tem ferramentas nem ações, e a saída é presa a um esquema.")


if __name__ == "__main__":
    main()
