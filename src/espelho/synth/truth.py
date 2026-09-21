"""Sorteia a verdade das reviews: quem escreveu, sobre o quê e com que nota.

O texto vem depois (dia 1, passo 4). Sortear a verdade primeiro garante que
sabemos a resposta certa de cada review e de cada evento plantado.

Ideia central: para cada review e para cada aspecto disponível no canal
(salão ou delivery) jogamos uma moeda com uma probabilidade. Os eventos
plantados simplesmente mudam essas probabilidades a partir de um certo dia.
"""

import json
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from espelho.schema import ASPECTS_BY_CHANNEL, POLARITIES

START_DATE = date(2026, 6, 1)
MAX_MENTIONS = 3  # uma review com mais de 3 assuntos vira texto confuso

STORES = [
    {"store_id": 1, "name": "Loja 1", "cuisine": "pizzaria", "delivery_share": 0.55},
    {"store_id": 2, "name": "Loja 2", "cuisine": "cantina italiana", "delivery_share": 0.35},
    {"store_id": 3, "name": "Loja 3", "cuisine": "hamburgueria", "delivery_share": 0.75},
    {"store_id": 4, "name": "Loja 4", "cuisine": "japonês", "delivery_share": 0.50},
    {"store_id": 5, "name": "Loja 5", "cuisine": "churrascaria", "delivery_share": 0.20},
    {"store_id": 6, "name": "Loja 6", "cuisine": "marmitaria", "delivery_share": 0.80},
]

# (probabilidade de elogio, probabilidade de reclamação) por review do canal.
BASE_RATES = {
    "atendimento": (0.22, 0.08),
    "tempo_espera_salao": (0.05, 0.10),
    "prazo_entrega": (0.10, 0.06),
    "condicao_entrega": (0.06, 0.07),
    "qualidade_comida": (0.35, 0.10),
    "preco_valor": (0.08, 0.08),
    "ambiente_limpeza": (0.15, 0.05),
    "resposta_canal": (0.02, 0.02),
}

# Quanto cada menção move a nota (em estrelas). São os efeitos "verdadeiros"
# que a regressão do dia 5 vai tentar recuperar.
NEGATIVE_EFFECT = {
    "atendimento": -1.0,
    "tempo_espera_salao": -0.9,
    "prazo_entrega": -1.1,
    "condicao_entrega": -1.0,
    "qualidade_comida": -1.5,
    "preco_valor": -0.6,
    "ambiente_limpeza": -0.8,
    "resposta_canal": -0.7,
}
POSITIVE_EFFECT = {aspect: 0.4 for aspect in BASE_RATES}
POSITIVE_EFFECT["qualidade_comida"] = 0.6

BASE_STARS = 3.9
STAR_NOISE = 0.35


@dataclass(frozen=True)
class Event:
    """Um evento plantado: muda a probabilidade de um aspecto a partir de um dia."""

    event_id: str
    description: str
    store_id: int | None  # None = todas as lojas
    start_day: int
    aspect: str
    polarity: str
    mode: str  # "set" troca a probabilidade; "add" soma a ela
    value: float
    dish: str | None = None  # prato citado nos elogios, se houver


EVENTS = [
    Event("E1", "Demora na entrega", 3, 30, "prazo_entrega", "negativo", "set", 0.30),
    Event("E2", "Equipe nova piora o atendimento", 5, 40, "atendimento", "negativo", "set", 0.28),
    Event(
        "E3", "Prato novo gera elogios", 2, 25, "qualidade_comida", "positivo", "set", 0.55,
        dish="risoto de camarão",
    ),
    Event("E4", "Aumento de preço na rede toda", None, 45, "preco_valor", "negativo", "add", 0.06),
    Event("E5", "Demora na resposta do WhatsApp", 1, 20, "resposta_canal", "negativo", "set", 0.15),
]

TONES = ["informal com gíria", "neutro", "curto e seco", "detalhado", "irônico"]
TONE_PROBS = [0.35, 0.30, 0.20, 0.10, 0.05]
LENGTH_BY_TONE = {
    "informal com gíria": "2 a 3 frases",
    "neutro": "2 a 3 frases",
    "curto e seco": "1 frase",
    "detalhado": "4 a 5 frases",
    "irônico": "2 a 3 frases",
}


def event_is_active(event: Event, store_id: int, day: int) -> bool:
    return day >= event.start_day and event.store_id in (None, store_id)


def _mention_probs(store_id, day, channel, store_factors):
    """Probabilidade de cada (aspecto, polaridade) neste dia, loja e canal."""
    probs = {}
    for aspect in ASPECTS_BY_CHANNEL[channel]:
        positive_rate, negative_rate = BASE_RATES[aspect]
        for polarity, base in (("positivo", positive_rate), ("negativo", negative_rate)):
            p = base * store_factors[(store_id, aspect, polarity)]
            for event in EVENTS:
                same_target = event.aspect == aspect and event.polarity == polarity
                if same_target and event_is_active(event, store_id, day):
                    p = event.value if event.mode == "set" else p + event.value
            probs[(aspect, polarity)] = min(max(p, 0.0), 0.95)
    return probs


def _sample_mentions(rng, probs):
    """Joga uma moeda por (aspecto, polaridade). Pode sair zero menções: é realista."""
    by_aspect = {}
    for (aspect, polarity), p in probs.items():
        if rng.random() < p:
            by_aspect.setdefault(aspect, []).append(polarity)

    mentions = []
    for aspect, polarities in by_aspect.items():
        # Se o mesmo aspecto saiu elogio e reclamação, sorteia um dos dois.
        polarity = polarities[0] if len(polarities) == 1 else str(rng.choice(polarities))
        mentions.append({"aspect": aspect, "polarity": polarity})

    if len(mentions) > MAX_MENTIONS:
        keep = sorted(rng.choice(len(mentions), size=MAX_MENTIONS, replace=False))
        mentions = [mentions[i] for i in keep]
    return mentions


def _stars(rng, mentions):
    """Nota latente (contínua, sem corte) e nota observada (inteira de 1 a 5)."""
    latent = BASE_STARS + rng.normal(0, STAR_NOISE)
    for mention in mentions:
        effects = NEGATIVE_EFFECT if mention["polarity"] == "negativo" else POSITIVE_EFFECT
        latent += effects[mention["aspect"]]
    observed = int(np.clip(np.round(latent), 1, 5))
    return float(latent), observed


def _dish(rng, store_id, day, mentions):
    for event in EVENTS:
        if event.dish and event_is_active(event, store_id, day):
            praised = any(
                m["aspect"] == event.aspect and m["polarity"] == event.polarity
                for m in mentions
            )
            if praised and rng.random() < 0.7:
                return event.dish
    return None


def generate_truth(seed: int = 42, n_days: int = 60, mean_per_day: float = 10.0) -> pd.DataFrame:
    """Gera a tabela de verdade: uma linha por review, ainda sem texto."""
    rng = np.random.default_rng(seed)

    # Cada loja é um pouco diferente das outras: 80% a 120% das taxas base.
    store_factors = {
        (store["store_id"], aspect, polarity): rng.uniform(0.8, 1.2)
        for store in STORES
        for aspect in BASE_RATES
        for polarity in POLARITIES
    }

    rows = []
    for day in range(n_days):
        for store in STORES:
            for _ in range(rng.poisson(mean_per_day)):
                channel = "delivery" if rng.random() < store["delivery_share"] else "salao"
                probs = _mention_probs(store["store_id"], day, channel, store_factors)
                mentions = _sample_mentions(rng, probs)
                latent, stars = _stars(rng, mentions)
                tone = str(rng.choice(TONES, p=TONE_PROBS))
                active = [e.event_id for e in EVENTS if event_is_active(e, store["store_id"], day)]
                rows.append(
                    {
                        "store_id": store["store_id"],
                        "store_name": store["name"],
                        "cuisine": store["cuisine"],
                        "day": day,
                        "date": (START_DATE + timedelta(days=day)).isoformat(),
                        "channel": channel,
                        "mentions": json.dumps(mentions, ensure_ascii=False),
                        "n_mentions": len(mentions),
                        "stars_latent": round(latent, 3),
                        "stars": stars,
                        "tone": tone,
                        "sarcastic": tone == "irônico",
                        "has_typos": bool(rng.random() < 0.25),
                        "length_hint": LENGTH_BY_TONE[tone],
                        "dish": _dish(rng, store["store_id"], day, mentions),
                        "active_events": ",".join(active),
                    }
                )

    df = pd.DataFrame(rows)
    df.insert(0, "review_id", [f"R{i:05d}" for i in range(len(df))])
    return df
