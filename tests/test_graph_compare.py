import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from helpers import assert_matches

from glaurlex.core.graph import community_leiden, node_stats, partition_to_map
from glaurlex.core.graph_compare import (
    BOTH,
    ONLY_A,
    ONLY_B,
    PROFILE_METRICS,
    RAREFACTION_METRICS,
    canonical_edges,
    community_crosstab,
    community_shared_nodes,
    compare_pair_edges,
    compare_pair_nodes,
    degree_ccdf,
    degree_frame,
    edge_weight_divergence,
    graph_profile,
    intersection_sizes,
    node_metric_map,
    node_metric_table,
    node_presence_matrix,
    pairwise_similarity_matrix,
    pairwise_similarity_table,
    partition_similarity,
    permutation_overlap_test,
    profile_table,
    rank_correlation,
    rarefied_profiles,
    set_summary,
    union_graph,
)


@pytest.fixture(scope="module")
def pair(group_graphs):
    return group_graphs["F"], group_graphs["M"]


@pytest.fixture(scope="module")
def levels(group_graphs):
    return {k: group_graphs[k] for k in ("A", "B", "C")}


def weighted(edges, directed=False):
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_weighted_edges_from(edges)
    return graph


# --- parity with the reference implementation --------------------------------


def test_set_comparisons_parity(pair, levels, reference):
    a, b = pair
    edges = compare_pair_edges(a, b)
    for key in ("only_a", "only_b", "common"):
        edges[key] = [list(e) for e in edges[key]]
    del edges["norm_weights_a"], edges["norm_weights_b"]

    assert_matches(compare_pair_nodes(a, b), reference["compare_pair_nodes"])
    assert_matches(edges, reference["compare_pair_edges"])
    assert_matches(edge_weight_divergence(a, b), reference["edge_weight_divergence"])
    assert_matches(node_presence_matrix(levels), reference["node_presence_matrix"])
    assert_matches(set_summary(levels), reference["set_summary"])
    assert_matches(pairwise_similarity_matrix(levels).to_numpy(), reference["pairwise_nodes"])
    assert_matches(
        pairwise_similarity_matrix(levels, on="edges").to_numpy(), reference["pairwise_edges"]
    )
    assert_matches(intersection_sizes(levels, top=5), reference["intersection_sizes"])


def test_profile_parity(pair, reference):
    a, b = pair
    profiles = {"F": graph_profile(a), "M": graph_profile(b)}

    assert_matches(profiles, reference["graph_profile"])
    assert_matches(profile_table(profiles), reference["profile_table_two"])
    assert_matches(degree_ccdf(a), reference["degree_ccdf_f"])


def test_node_metric_parity(pair, reference):
    stats_a, stats_b = node_stats(pair[0]), node_stats(pair[1])
    table = node_metric_table({"F": stats_a, "M": stats_b}, "pagerank")
    correlation = rank_correlation(
        node_metric_map(stats_a, "pagerank"), node_metric_map(stats_b, "pagerank")
    )

    assert_matches(table, reference["node_metric_table_pagerank"])
    assert_matches(correlation, reference["rank_correlation_pagerank"])


def test_community_comparison_parity(pair, reference):
    part_a, part_b = (partition_to_map(community_leiden(g)) for g in pair)
    crosstab = community_crosstab(part_a, part_b)

    assert_matches({"F": part_a, "M": part_b}, reference["partitions"])
    assert_matches(partition_similarity(part_a, part_b), reference["partition_similarity"])
    assert_matches(
        {
            "index": list(crosstab.index),
            "columns": list(crosstab.columns),
            "values": crosstab.to_numpy(),
        },
        reference["community_crosstab"],
    )


def test_union_graph_parity(pair, reference):
    union = union_graph({"F": pair[0], "M": pair[1]})

    assert_matches(dict(union.nodes(data=True)), reference["union_nodes"])
    assert_matches({f"{u}|{v}": d for u, v, d in union.edges(data=True)}, reference["union_edges"])


def test_resampling_parity(ids_by_group, build_graph, reference):
    ids = {"F": ids_by_group["F"], "M": ids_by_group["M"]}
    rarefied = rarefied_profiles(build_graph, ids, k=3, seed=42)
    expected = dict(reference["rarefied_profiles"])

    assert (rarefied.attrs["n_min"], rarefied.attrs["k"]) == (
        expected.pop("n_min"),
        expected.pop("k"),
    )
    assert_matches(rarefied, expected)
    assert_matches(
        permutation_overlap_test(build_graph, ids["F"], ids["M"], k=20, seed=7),
        reference["permutation_overlap"],
    )


# --- behaviour on hand-built graphs -------------------------------------------


def test_canonical_edges_merge_only_undirected_reversals():
    edges = [("b", "a", 2.0), ("c", "d", 1.0)]

    assert canonical_edges(weighted(edges)) == {("a", "b"): 2.0, ("c", "d"): 1.0}
    assert canonical_edges(weighted([*edges, ("a", "b", 5.0)], directed=True)) == {
        ("b", "a"): 2.0,
        ("c", "d"): 1.0,
        ("a", "b"): 5.0,
    }


def test_pair_indices_on_known_sets():
    a = weighted([("a", "b", 1), ("b", "c", 1), ("c", "d", 1)])
    b = weighted([("b", "c", 1), ("c", "d", 1), ("d", "e", 1)])
    out = compare_pair_nodes(a, b)

    assert (out["only_a"], out["only_b"], out["common"]) == (["a"], ["e"], ["b", "c", "d"])
    assert (out["jaccard"], out["overlap_coef"], out["dice"]) == pytest.approx(
        (3 / 5, 3 / 4, 6 / 8)
    )


def test_edge_weight_statistics_need_three_shared_edges():
    a = weighted([("a", "b", 1), ("b", "c", 3)])
    b = weighted([("a", "b", 2), ("b", "c", 6), ("c", "d", 1)])
    sparse = compare_pair_edges(a, b)
    assert math.isnan(sparse["weight_spearman"])

    a.add_edge("c", "d", weight=2)
    dense = compare_pair_edges(a, b)
    assert dense["n_common"] == 3
    assert dense["weight_cosine"] == pytest.approx(
        np.dot([1, 3, 2], [2, 6, 1]) / (np.linalg.norm([1, 3, 2]) * np.linalg.norm([2, 6, 1]))
    )


def test_divergence_marks_presence_and_sorts_by_magnitude():
    out = edge_weight_divergence(
        weighted([("a", "b", 3), ("b", "c", 1)]), weighted([("b", "c", 1), ("c", "d", 1)])
    )

    assert out["presence"].tolist() == [ONLY_A, ONLY_B, BOTH]
    assert out["delta"].abs().is_monotonic_decreasing
    assert out["delta"].sum() == pytest.approx(0.0)


def test_presence_summary_and_intersections_agree():
    graphs = {
        "x": weighted([("a", "b", 1)]),
        "y": weighted([("b", "c", 1)]),
        "z": weighted([("b", "d", 1)]),
    }
    presence = node_presence_matrix(graphs).set_index("node")
    summary = set_summary(graphs).set_index("entity")
    sizes = intersection_sizes(graphs).set_index("combination")

    assert presence.loc["b", "n_graphs"] == 3
    assert presence.loc["a", "exclusive_to"] == "x"
    assert presence.loc["b", "exclusive_to"] is None
    assert summary["exclusive"].tolist() == [1, 1, 1]
    assert summary["in_all"].tolist() == [1, 1, 1]
    assert sizes.loc["x ∩ y ∩ z", "n_nodes"] == 1
    assert sizes["n_nodes"].sum() == len(presence)


def test_pairwise_similarity_table_matches_matrices(levels):
    table = pairwise_similarity_table(levels)
    nodes = pairwise_similarity_matrix(levels)

    assert len(table) == 3
    assert table["jaccard_nodes"].is_monotonic_decreasing
    for _, row in table.iterrows():
        assert row["jaccard_nodes"] == nodes.loc[row["entity_a"], row["entity_b"]]
    with pytest.raises(ValueError, match="on must be"):
        pairwise_similarity_matrix(levels, on="types")


def test_profile_accepts_full_node_stats_and_external_partition(pair):
    graph = pair[0]
    light = graph_profile(graph)
    full = graph_profile(graph, node_stats(graph), partition_to_map(community_leiden(graph)))

    assert light == pytest.approx(full, nan_ok=True)
    assert set(light) == set(PROFILE_METRICS)
    assert np.isnan(graph_profile(graph, community_map={"w01": 1})["modularity"])


def test_profile_of_disconnected_and_empty_graphs():
    profile = graph_profile(weighted([("a", "b", 1), ("b", "c", 1), ("x", "y", 1)]))
    assert profile["largest_component_fraction"] == pytest.approx(3 / 5)
    assert profile["components"] == 2

    empty = graph_profile(nx.Graph())
    assert (empty["n_communities"], empty["n_nodes"]) == (0, 0)
    assert math.isnan(empty["modularity"])


def test_profile_table_deltas_only_for_two_entities():
    two = profile_table({"a": {"n_nodes": 10}, "b": {"n_nodes": 15}}).set_index("metric")
    assert two.loc["n_nodes", ["delta", "delta_pct"]].tolist() == [5.0, 50.0]
    assert "delta" not in profile_table({"a": {}, "b": {}, "c": {}}).columns


def test_degree_ccdf_on_star():
    out = degree_ccdf(nx.star_graph(4))

    assert out["degree"].tolist() == [1.0, 4.0]
    assert out["ccdf"].tolist() == [1.0, 0.2]


def test_node_metric_table_keeps_missing_nodes_as_nan():
    a = pd.DataFrame({"node": ["x", "y"], "pagerank": [0.6, 0.4]})
    b = pd.DataFrame({"node": ["y", "z"], "pagerank": [0.9, 0.1]})
    out = node_metric_table({"A": a, "B": b}, "pagerank").set_index("node")

    assert out.loc["y", ["delta_value", "delta_rank", "presence"]].tolist() == [
        pytest.approx(0.5),
        1,
        BOTH,
    ]
    assert np.isnan(out.loc["z", "A"])
    assert out.loc["x", "presence"] == ONLY_A
    with pytest.raises(KeyError, match="betweenness"):
        node_metric_table({"A": a}, "betweenness")


def test_rank_correlation_direction_and_guard():
    a = {"p": 1.0, "q": 2.0, "r": 3.0, "s": 4.0}

    assert rank_correlation(a, {k: -v for k, v in a.items()})["spearman_rho"] == pytest.approx(-1.0)
    assert rank_correlation(a, {"p": 1.0, "q": 1.0, "r": 1.0})["error"] == "insufficient_data"


@pytest.mark.parametrize(
    ("part_b", "ari", "nmi"),
    [
        ({1: "x", 2: "x", 3: "y", 4: "y"}, 1.0, 1.0),
        ({1: 0, 2: 1, 3: 0, 4: 1}, -0.5, 0.0),
    ],
    ids=["relabelled", "independent"],
)
def test_partition_similarity_known_values(part_b, ari, nmi):
    out = partition_similarity({1: 0, 2: 0, 3: 1, 4: 1, 99: 5}, part_b)

    assert out["n_common"] == 4
    assert (out["ari"], out["nmi"]) == pytest.approx((ari, nmi))


def test_partition_similarity_degenerate_cases():
    trivial = partition_similarity({1: 0, 2: 0}, {1: "a", 2: "a"})
    assert (trivial["ari"], trivial["nmi"]) == (1.0, 1.0)
    assert math.isnan(partition_similarity({1: 0}, {1: 0, 2: 1})["ari"])


def test_shared_nodes_explain_the_crosstab():
    part_a = {"a": 1, "b": 1, "c": 2, "only_a": 3}
    part_b = {"a": "x", "b": "y", "c": "y", "only_b": "z"}
    crosstab = community_crosstab(part_a, part_b)
    shared = community_shared_nodes(part_a, part_b)

    assert shared == {(1, "x"): ["a"], (1, "y"): ["b"], (2, "y"): ["c"]}
    assert all(crosstab.loc[ca, cb] == len(nodes) for (ca, cb), nodes in shared.items())
    assert community_crosstab({"a": 1}, {"b": 2}).empty


def test_union_graph_round_trips_through_gml(tmp_path):
    graphs = {"Group A/1": weighted([("a", "b", 2)]), "Group A 1": weighted([("b", "c", 1)])}
    union = union_graph(graphs)
    path = tmp_path / "union.gml"
    nx.write_gml(union, path)
    loaded = nx.read_gml(path)

    assert loaded.nodes["b"] == {
        "n_graphs": 2,
        "membership": BOTH,
        "in_Group_A_1": 1,
        "in_Group_A_1_1": 1,
    }
    assert loaded.edges["a", "b"]["weight"] == pytest.approx(0.5)


def test_union_graph_direction_and_membership_labels():
    directed = weighted([("a", "b", 1)], directed=True)
    assert union_graph({"x": directed, "y": directed}).is_directed()
    assert not union_graph({"x": directed, "y": weighted([("a", "b", 1)])}).is_directed()

    three = union_graph(
        {"x": weighted([("a", "b", 1)]), "y": nx.Graph(), "z": weighted([("a", "c", 1)])}
    )
    assert three.nodes["a"]["membership"] == "2/3"
    with pytest.raises(ValueError, match="at least one graph"):
        union_graph({})


def test_rarefaction_equalizes_sizes_and_is_reproducible(build_graph, ids_by_group):
    ids = {"F": ids_by_group["F"][:6], "M": ids_by_group["M"]}
    calls = []
    first = rarefied_profiles(
        build_graph, ids, k=4, seed=3, progress_cb=lambda i, n: calls.append((i, n))
    )
    second = rarefied_profiles(build_graph, ids, k=4, seed=3)

    pd.testing.assert_frame_equal(first, second)
    assert first.attrs["n_min"] == 6
    assert first["metric"].tolist() == list(RAREFACTION_METRICS)
    assert first.set_index("metric").loc["n_nodes", "F (sd)"] == 0.0
    assert calls[-1] == (8, 8)


def test_permutation_test_detects_separated_groups():
    vocab = {1: "abcd", 2: "abce", 3: "abdf", 4: "mnop", 5: "mnoq", 6: "mnpr"}

    def build(ids):
        graph = nx.Graph()
        for i in ids:
            nx.add_path(graph, vocab[i])
        return graph

    out = permutation_overlap_test(build, [1, 2, 3], [4, 5, 6], k=50, seed=0)
    assert out["jaccard_nodes"]["observed"] == 0.0
    assert out["jaccard_nodes"]["p_value"] < 0.1
    assert len(out["jaccard_edges"]["null"]) == 50


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda b: rarefied_profiles(b, {"x": [1]}, k=0), "k must be"),
        (lambda b: rarefied_profiles(b, {}), "at least one label"),
        (lambda b: permutation_overlap_test(b, [1], [2], k=0), "k must be"),
        (lambda b: permutation_overlap_test(b, [], [2]), "at least one informant"),
    ],
)
def test_resampling_rejects_invalid_arguments(build_graph, call, message):
    with pytest.raises(ValueError, match=message):
        call(build_graph)


def test_degree_frame_matches_node_stats(pair):
    light = degree_frame(pair[0]).set_index("node").sort_index()
    full = node_stats(pair[0]).set_index("node")[["degree", "strength"]].sort_index()

    pd.testing.assert_frame_equal(light, full)
