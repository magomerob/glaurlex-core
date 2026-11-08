import dataclasses
import importlib

import pytest

from glaurlex.core.graph import NODE_STATS_COLUMNS
from glaurlex.core.inference import informant_metrics
from glaurlex.core.metrics_catalog import (
    METRIC_CATALOG,
    MetricSpec,
    labels_by_scope,
    metrics_by_scope,
)
from glaurlex.core.stats import TYPE_STATISTICS_COLUMNS


@pytest.mark.parametrize("key", list(METRIC_CATALOG))
def test_implementation_resolves_to_a_callable(key):
    module, _, name = METRIC_CATALOG[key].implementation.rpartition(".")

    assert callable(getattr(importlib.import_module(module), name))


def test_keys_and_names_are_unique_and_scoped():
    specs = list(METRIC_CATALOG.values())

    assert all(key == spec.key for key, spec in METRIC_CATALOG.items())
    assert len({spec.name for spec in specs}) == len(specs)
    assert {spec.scope for spec in specs} == {"informant", "type", "node", "comparison"}
    assert labels_by_scope("node") == {k: s.name for k, s in metrics_by_scope("node").items()}
    with pytest.raises(dataclasses.FrozenInstanceError):
        specs[0].key = "x"  # type: ignore[misc]


def test_catalog_keys_name_real_output_columns(topic, informants):
    produced = {
        "informant": set(informant_metrics(topic, informants).columns),
        "type": set(TYPE_STATISTICS_COLUMNS),
        "node": set(NODE_STATS_COLUMNS),
    }
    for scope, columns in produced.items():
        assert set(metrics_by_scope(scope)) <= columns, scope


def test_metric_spec_defaults():
    spec = MetricSpec("k", "Name", "Description.", "node", "Algorithm.")

    assert (spec.parameters, spec.implementation) == ("", "")
