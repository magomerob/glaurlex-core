import numpy as np
import pandas as pd
import pytest

from glaurlex.core.variables import get_order, is_numeric_like, is_ordinal, split_variables

CONFIG = {
    "LEVEL": {"ordinal": True, "order": ["primary", "secondary", "tertiary"]},
    "SEX": {"ordinal": False, "order": ["F", "M"]},
    "EMPTY": {"ordinal": True, "order": []},
}


@pytest.mark.parametrize(
    ("col", "order"),
    [("LEVEL", ["primary", "secondary", "tertiary"]), ("SEX", []), ("EMPTY", []), ("MISSING", [])],
)
def test_order_and_ordinality(col, order):
    assert get_order(CONFIG, col) == order
    assert is_ordinal(CONFIG, col) is bool(order)


def test_split_variables_requires_parseable_and_many_distinct_values():
    df = pd.DataFrame(
        {
            "AGE": np.arange(20, 40),
            "AGE_TEXT": [str(x) for x in range(20, 39)] + ["unknown"],
            "SCORE": [1, 2, 3, 4] * 5,
            "MOSTLY_TEXT": ["a"] * 3 + list(range(17)),
            "REGION": ["north", "south"] * 10,
        }
    )
    categorical, numeric = split_variables(df, [*df.columns, "ABSENT"])

    assert numeric == ["AGE", "AGE_TEXT"]
    assert categorical == ["MOSTLY_TEXT", "REGION", "SCORE"]
    assert split_variables(df, ["AGE"], min_unique=21) == (["AGE"], [])


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.5, None, 3.0], True),
        (["1", "2", "x"], False),
        (["1"] * 9 + ["x"], True),
        (["a", "b"], False),
    ],
)
def test_is_numeric_like(values, expected):
    assert is_numeric_like(pd.Series(values)) is expected
