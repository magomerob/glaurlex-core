"""Fixtures built from ``tests/data``.

``topic.csv`` and ``informants.csv`` hold a synthetic topic (48 informants,
40 types). ``reference.json`` holds the outputs of the pre-split GLAURLex
implementation on that data, used as the parity contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from glaurlex.core.graph import build_cooccurrence_graph

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def reference() -> dict[str, Any]:
    return json.loads((DATA / "reference.json").read_text())


@pytest.fixture(scope="session")
def topic() -> pd.DataFrame:
    return pd.read_csv(DATA / "topic.csv")


@pytest.fixture(scope="session")
def informants() -> pd.DataFrame:
    return pd.read_csv(DATA / "informants.csv")


@pytest.fixture(scope="session")
def ids_by_group(informants: pd.DataFrame) -> dict[str, list[int]]:
    groups = {
        "F": ("SEX", "F"),
        "M": ("SEX", "M"),
        "A": ("LEVEL", "A"),
        "B": ("LEVEL", "B"),
        "C": ("LEVEL", "C"),
    }
    return {
        name: informants.loc[informants[col] == value, "informant_id"].tolist()
        for name, (col, value) in groups.items()
    }


@pytest.fixture(scope="session")
def build_graph(topic: pd.DataFrame):
    def build(ids):
        return build_cooccurrence_graph(topic[topic["informant_id"].isin(set(ids))], directed=False)

    return build


@pytest.fixture(scope="session")
def group_graphs(ids_by_group, build_graph):
    return {name: build_graph(ids) for name, ids in ids_by_group.items()}
