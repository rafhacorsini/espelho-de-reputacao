import json

import pytest

from espelho.schema import ASPECTS_BY_CHANNEL
from espelho.synth.checks import event_check
from espelho.synth.truth import MAX_MENTIONS, generate_truth


@pytest.fixture(scope="module")
def truth():
    return generate_truth(seed=42)


def test_same_seed_gives_same_data():
    first = generate_truth(seed=7, n_days=5)
    second = generate_truth(seed=7, n_days=5)
    assert first.equals(second)


def test_different_seed_gives_different_data():
    first = generate_truth(seed=1, n_days=5)
    second = generate_truth(seed=2, n_days=5)
    assert not first["mentions"].equals(second["mentions"])


def test_stars_and_mention_limits(truth):
    assert truth["stars"].between(1, 5).all()
    assert (truth["n_mentions"] <= MAX_MENTIONS).all()


def test_aspects_respect_the_channel(truth):
    for channel, text in zip(truth["channel"], truth["mentions"]):
        for mention in json.loads(text):
            assert mention["aspect"] in ASPECTS_BY_CHANNEL[channel]


def test_planted_events_are_visible(truth):
    check = event_check(truth).set_index("evento")
    # cada evento tem que subir bem acima do nível anterior
    assert check.loc["E1", "depois"] > check.loc["E1", "antes"] + 0.10
    assert check.loc["E2", "depois"] > check.loc["E2", "antes"] + 0.10
    assert check.loc["E3", "depois"] > check.loc["E3", "antes"] + 0.10
    assert check.loc["E4", "depois"] > check.loc["E4", "antes"] + 0.02
    assert check.loc["E5", "depois"] > check.loc["E5", "antes"] + 0.06
