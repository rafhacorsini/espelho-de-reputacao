"""Trava de sanidade sobre o conjunto ouro: roda depois de scripts/split_gold.py."""

from pathlib import Path

import pandas as pd
import pytest

DEV = Path("data/gold/dev.parquet")
TEST = Path("data/gold/test.parquet")

pytestmark = pytest.mark.skipif(
    not (DEV.exists() and TEST.exists()), reason="rode scripts/split_gold.py primeiro"
)


def test_sizes():
    assert len(pd.read_parquet(DEV)) == 100
    assert len(pd.read_parquet(TEST)) == 50


def test_dev_and_test_do_not_overlap():
    dev_ids = set(pd.read_parquet(DEV)["review_id"])
    test_ids = set(pd.read_parquet(TEST)["review_id"])
    assert dev_ids.isdisjoint(test_ids)


def test_no_empty_text_or_missing_labels():
    for path in (DEV, TEST):
        df = pd.read_parquet(path)
        assert (df["text"].str.len() > 0).all()
        assert df["gold_mentions"].notna().all()
