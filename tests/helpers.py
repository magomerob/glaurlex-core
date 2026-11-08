from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# PageRank, eigenvector centrality and Shapiro-Wilk differ across the supported
# scipy/networkx versions and CPUs: up to ~2e-5 relative (Shapiro-Wilk p-value,
# scipy 1.11 on x86_64).
RTOL = 5e-5
ATOL = 1e-10


def _same(actual: Any, expected: Any) -> bool:
    if expected is None or (isinstance(expected, float) and math.isnan(expected)):
        return (
            actual is None or actual is pd.NA or (isinstance(actual, float) and math.isnan(actual))
        )
    if isinstance(expected, bool) or isinstance(actual, (bool, np.bool_)):
        return bool(actual) == expected
    if isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=RTOL, abs_tol=ATOL)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(map(_same, actual, expected))
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same(actual[k], expected[k]) for k in expected
        )
    return actual == expected


def to_plain(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {
            "columns": list(value.columns),
            "data": {c: to_plain(list(value[c])) for c in value},
        }
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [to_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def assert_matches(actual: Any, expected: Any) -> None:
    """Compare an output with its reference, numbers within RTOL/ATOL."""
    plain = to_plain(actual)
    if isinstance(expected, dict) and isinstance(plain, dict):
        assert plain.keys() == expected.keys()
        for key, value in expected.items():
            assert _same(plain[key], value), f"{key}: {plain[key]!r} != {value!r}"
    else:
        assert _same(plain, expected), f"{plain!r} != {expected!r}"
