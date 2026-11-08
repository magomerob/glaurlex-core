"""Informant groups defined by filter rules over sociolinguistic variables."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class Group:
    """A named set of filter rules over the informants frame.

    Attributes
    ----------
    name : str
    filters : mapping
        ``column -> rule``. A rule is a list (membership), a scalar (equality)
        or a mapping in one of two shapes:

        - ``{op: value, ...}`` with ``op`` in ``"<=" "<" ">=" ">" "==" "!="``;
          all comparisons must hold;
        - ``{"op": op, "value": value}`` with the same operators, or
          ``{"op": "between", "min": low, "max": high}`` (bounds inclusive).

        Rules on different columns are combined with AND.
    """

    name: str
    filters: Mapping[str, Any] = field(default_factory=dict)


_COMPARISONS = {
    "<=": operator.le,
    "<": operator.lt,
    ">=": operator.ge,
    ">": operator.gt,
    "==": operator.eq,
    "!=": operator.ne,
}


def _comparisons(rule: Mapping[str, Any]) -> list[tuple[str, Any]]:
    if "op" not in rule:
        pairs = list(rule.items())
    else:
        op = rule["op"]
        keys = {"op", "min", "max"} if op == "between" else {"op", "value"}
        if set(rule) != keys:
            msg = f"rule {dict(rule)!r} must have exactly the keys {sorted(keys)}"
            raise ValueError(msg)
        if op == "between":
            return [(op, (rule["min"], rule["max"]))]
        pairs = [(op, rule["value"])]
    for op, _ in pairs:
        if op not in _COMPARISONS:
            msg = f"unknown operator {op!r} in rule {dict(rule)!r}"
            raise ValueError(msg)
    return pairs


def apply_group(informants: pd.DataFrame, group: Group) -> pd.DataFrame:
    """Return the informants that satisfy every rule of a group.

    Parameters
    ----------
    informants : pandas.DataFrame
        Informants frame containing every column referenced by ``group.filters``.
    group : Group

    Returns
    -------
    pandas.DataFrame
        Filtered copy of ``informants``.

    Raises
    ------
    ValueError
        If a mapping rule has an unknown operator, or an ``"op"`` rule lacks
        or adds keys.
    """
    df = informants.copy()
    for col, rule in group.filters.items():
        if isinstance(rule, list):
            df = df[df[col].isin(rule)]
        elif isinstance(rule, Mapping):
            for op, value in _comparisons(rule):
                if op == "between":
                    df = df[df[col].between(*value)]
                else:
                    df = df[_COMPARISONS[op](df[col], value)]
        else:
            df = df[df[col] == rule]
    return df


def select_informant_rows(df: pd.DataFrame, informants: pd.DataFrame) -> pd.DataFrame:
    """Keep the responses of the informants listed in ``informants``.

    Parameters
    ----------
    df : pandas.DataFrame
        Topic frame with an ``informant_id`` column.
    informants : pandas.DataFrame
        Informants frame, typically the output of :func:`apply_group`.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        If either frame lacks an ``informant_id`` column.
    """
    for name, frame in (("df", df), ("informants", informants)):
        if "informant_id" not in frame.columns:
            msg = f"{name} must have an 'informant_id' column"
            raise ValueError(msg)
    return df[df["informant_id"].isin(informants["informant_id"])]
