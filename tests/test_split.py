from pathlib import Path

import pandas as pd
import pytest

from recclens.data.split import split_by_group

FIXTURES = Path(__file__).parent / "fixtures"


def test_groups_never_cross_splits():
    df = pd.read_csv(FIXTURES / "sample_items.csv")
    parts = split_by_group(df, [0.5, 0.25, 0.25], seed=0)

    group_sets = [set(p["label_group"]) for p in parts.values()]
    for i, a in enumerate(group_sets):
        for b in group_sets[i + 1 :]:
            assert not a & b
    assert sum(len(p) for p in parts.values()) == len(df)


def test_split_is_deterministic():
    df = pd.read_csv(FIXTURES / "sample_items.csv")
    a = split_by_group(df, [0.5, 0.25, 0.25], seed=7)
    b = split_by_group(df, [0.5, 0.25, 0.25], seed=7)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_ratios_must_sum_to_one():
    df = pd.read_csv(FIXTURES / "sample_items.csv")
    with pytest.raises(ValueError):
        split_by_group(df, [0.5, 0.5, 0.5], seed=0)
