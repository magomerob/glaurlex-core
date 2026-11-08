"""Classification of informant variables and ordinal level configuration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

VariablesConfig = Mapping[str, Mapping[str, Any]]
"""``column -> {"ordinal": bool, "order": list of level labels}``."""


def get_order(config: VariablesConfig, col: str) -> list[str]:
    """Return the declared level order of an ordinal column.

    Parameters
    ----------
    config : mapping
        ``column -> {"ordinal": bool, "order": list}``.
    col : str

    Returns
    -------
    list of str
        Empty when the column is absent or not marked ordinal.
    """
    cfg = config.get(col) or {}
    if not cfg.get("ordinal"):
        return []
    return list(cfg.get("order") or [])


def is_ordinal(config: VariablesConfig, col: str) -> bool:
    """Return whether a column is ordinal with at least one declared level.

    Parameters
    ----------
    config : mapping
    col : str

    Returns
    -------
    bool
    """
    return bool(get_order(config, col))


def is_numeric_like(series: pd.Series, min_ratio: float = 0.9) -> bool:
    """Return whether a series can be treated as numeric.

    Parameters
    ----------
    series : pandas.Series
    min_ratio : float, default 0.9
        Minimum share of values that must parse as numbers when the dtype is
        not already numeric.

    Returns
    -------
    bool
    """
    if pd.api.types.is_numeric_dtype(series):
        return True
    return bool(pd.to_numeric(series, errors="coerce").notna().mean() >= min_ratio)


def split_variables(
    df: pd.DataFrame,
    candidates: Iterable[str],
    *,
    min_unique: int = 8,
    min_ratio: float = 0.9,
) -> tuple[list[str], list[str]]:
    """Split columns into categorical and continuous numeric variables.

    Parameters
    ----------
    df : pandas.DataFrame
    candidates : iterable of str
        Columns to classify; names absent from ``df`` are skipped.
    min_unique : int, default 8
        Minimum distinct numeric values for a column to count as continuous.
    min_ratio : float, default 0.9
        Minimum share of values that must parse as numbers.

    Returns
    -------
    categorical, numeric : list of str
        Both sorted alphabetically. A column is numeric when at least
        ``min_ratio`` of its values parse as numbers and those values take at
        least ``min_unique`` distinct values; otherwise it is categorical.
    """
    categorical: list[str] = []
    numeric: list[str] = []
    for col in candidates:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        continuous = values.notna().mean() >= min_ratio and values.nunique() >= min_unique
        (numeric if continuous else categorical).append(col)
    return sorted(categorical), sorted(numeric)
