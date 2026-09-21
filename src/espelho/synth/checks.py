"""Checagens de sanidade da verdade sorteada."""

import json

import pandas as pd

from espelho.schema import ASPECTS_BY_CHANNEL
from espelho.synth.truth import EVENTS


def event_check(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada evento plantado: a taxa antes e depois do dia de início.

    A taxa é a fração das reviews elegíveis (loja e canal certos) que mencionam
    o aspecto do evento com a polaridade do evento. Se "depois" não for bem
    maior que "antes", o gerador está errado.
    """
    rows = []
    for event in EVENTS:
        eligible = df if event.store_id is None else df[df["store_id"] == event.store_id]
        eligible = eligible[eligible["channel"].map(lambda c: event.aspect in ASPECTS_BY_CHANNEL[c])]
        has_mention = eligible["mentions"].map(
            lambda text: any(
                m["aspect"] == event.aspect and m["polarity"] == event.polarity
                for m in json.loads(text)
            )
        )
        rows.append(
            {
                "evento": event.event_id,
                "loja": "todas" if event.store_id is None else event.store_id,
                "dia_inicio": event.start_day,
                "aspecto": f"{event.aspect} ({event.polarity})",
                "antes": has_mention[eligible["day"] < event.start_day].mean(),
                "depois": has_mention[eligible["day"] >= event.start_day].mean(),
            }
        )
    return pd.DataFrame(rows)
