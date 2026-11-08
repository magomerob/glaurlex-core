import pandas as pd
import pytest
from helpers import assert_matches

from glaurlex.core.stats import TYPE_STATISTICS_COLUMNS, type_statistics


def topic_frame(*lists: str) -> pd.DataFrame:
    rows = [(i, p, t) for i, words in enumerate(lists, start=1) for p, t in enumerate(words)]
    return pd.DataFrame(rows, columns=["informant_id", "pos", "type"])


# Reference availability values computed with LexPro.
@pytest.mark.parametrize(
    ("lists", "expected"),
    [
        (("abc", "bac", "bca"), {"b": 0.772212, "a": 0.472299, "c": 0.172385}),
        (("ab", "ac", "ad"), {"a": 1.0, "b": 0.0334196, "c": 0.0334196, "d": 0.0334196}),
    ],
)
def test_availability_matches_lexpro(lists, expected):
    out = type_statistics(topic_frame(*lists)).set_index("type")["availability"]
    assert out.to_dict() == pytest.approx(expected, abs=1e-4)


def test_columns_are_mutually_consistent():
    out = type_statistics(topic_frame("abcd", "bae", "cfab", "a"))

    assert tuple(out.columns) == TYPE_STATISTICS_COLUMNS
    assert out["availability"].is_monotonic_decreasing
    assert out["cum_freq"].iloc[-1] == pytest.approx(1.0)
    assert out.set_index("type").loc["a", ["tokens", "appearance", "avg_pos"]].tolist() == [
        4,
        1.0,
        0.75,
    ]
    assert out.set_index("type").loc["f", "appearance"] == pytest.approx(0.25)


def test_single_position_topic_gives_unit_availability():
    out = type_statistics(topic_frame("a", "a", "b")).set_index("type")

    assert (out["availability"] == 1.0).all()
    assert (out["avg_pos"] == 0.0).all()
    assert out["appearance"].to_dict() == pytest.approx({"a": 2 / 3, "b": 1 / 3})


def test_informant_filter_equals_prefiltered_frame(topic, informants):
    ids = informants.loc[informants["LEVEL"] == "B", "informant_id"]
    filtered = type_statistics(topic, informant_ids=ids)
    prefiltered = type_statistics(topic[topic["informant_id"].isin(ids)])

    pd.testing.assert_frame_equal(filtered, prefiltered)


def test_ties_are_ordered_by_type():
    topic = pd.DataFrame(
        {"informant_id": [1, 2, 3, 4], "pos": [0, 0, 0, 0], "type": ["c", "a", "b", "a"]}
    )

    assert type_statistics(topic)["type"].tolist() == ["a", "b", "c"]


@pytest.mark.parametrize("frame", [topic_frame(), topic_frame("ab")], ids=["empty", "filtered-out"])
def test_empty_input_keeps_columns(frame):
    out = type_statistics(frame, informant_ids=[] if len(frame) else None)

    assert out.empty
    assert tuple(out.columns) == TYPE_STATISTICS_COLUMNS


@pytest.mark.parametrize(
    ("key", "level"), [("type_statistics", None), ("type_statistics_level_b", "B")]
)
def test_parity_with_reference(topic, informants, reference, key, level):
    ids = None if level is None else informants.loc[informants["LEVEL"] == level, "informant_id"]
    assert_matches(type_statistics(topic, informant_ids=ids).reset_index(drop=True), reference[key])
