"""Informant groups defined by filter rules over sociolinguistic variables."""

from __future__ import annotations

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
        ``column -> rule``. A rule is a list (membership), a mapping from
        comparison operator (``"<="``, ``">="``, ``"!="``, ``"=="``) to value,
        or a scalar (equality). Rules on different columns are combined with AND.
    """

    name: str
    filters: Mapping[str, Any] = field(default_factory=dict)


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

    Notes
    -----
    Mapping rules only recognize the keys ``"<="``, ``">="``, ``"!="`` and
    ``"=="``; any other key is ignored and does not filter.
    """
    df = informants.copy()
    for col, rule in group.filters.items():
        if isinstance(rule, list):
            df = df[df[col].isin(rule)]
        elif isinstance(rule, Mapping):
            for op, value in rule.items():
                if op == "<=":
                    df = df[df[col] <= value]
                elif op == ">=":
                    df = df[df[col] >= value]
                elif op == "!=":
                    df = df[df[col] != value]
                elif op == "==":
                    df = df[df[col] == value]
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
