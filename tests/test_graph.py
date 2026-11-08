import networkx as nx
import pandas as pd
import pytest
from helpers import assert_matches

from glaurlex.core.graph import (
    DIRECTED_NODE_STATS_COLUMNS,
    NODE_STATS_COLUMNS,
    SmallWorld,
    bigrams_for_topic,
    bigrams_to_dirgraph,
    bigrams_to_undgraph,
    bigrams_to_unordered,
    build_cooccurrence_graph,
    community_leiden,
    connected_components_sorted,
    graph_stats,
    is_graph_connected,
    node_stats,
    partition_to_map,
    small_world_indices,
)


def topic_frame(rows):
    return pd.DataFrame(rows, columns=["informant_id", "pos", "type"])


def as_records(df):
    return sorted(map(tuple, df[["type_1", "type_2", "count"]].to_numpy().tolist()))


def test_bigrams_follow_position_order_within_each_informant():
    df = topic_frame([(1, 2, "c"), (1, 0, "a"), (1, 1, "b"), (2, 0, "b"), (2, 1, "c"), (3, 0, "z")])

    assert as_records(bigrams_for_topic(df)) == [("a", "b", 1), ("b", "c", 2)]


def test_bigrams_keep_direction():
    df = topic_frame([(1, 0, "a"), (1, 1, "b"), (2, 0, "b"), (2, 1, "a")])

    assert as_records(bigrams_for_topic(df)) == [("a", "b", 1), ("b", "a", 1)]


@pytest.mark.parametrize(
    "rows", [[], [(1, 0, "a"), (2, 0, "b")]], ids=["empty", "single-token-lists"]
)
def test_bigrams_without_pairs(rows):
    out = bigrams_for_topic(topic_frame(rows))

    assert out.empty
    assert list(out.columns) == ["type_1", "type_2", "count"]


def test_unordered_bigrams_merge_both_directions():
    df = pd.DataFrame(
        {"type_1": ["b", "a", "c", "d"], "type_2": ["a", "b", "d", "c"], "count": [3, 2, 1, 4]}
    )

    assert as_records(bigrams_to_unordered(df)) == [("a", "b", 5), ("c", "d", 5)]


def test_graphs_carry_counts_as_weights():
    df = pd.DataFrame({"type_1": ["a", "b", "a"], "type_2": ["b", "a", "c"], "count": [2, 3, 4]})
    directed, undirected = bigrams_to_dirgraph(df), bigrams_to_undgraph(df)

    assert isinstance(directed, nx.DiGraph)
    assert {(u, v): w for u, v, w in directed.edges(data="weight")} == {
        ("a", "b"): 2,
        ("b", "a"): 3,
        ("a", "c"): 4,
    }
    assert not undirected.is_directed()
    assert undirected["b"]["a"]["weight"] == 5
    assert undirected.number_of_edges() == 2


def test_build_cooccurrence_graph_composes_bigram_steps(topic):
    bigrams = bigrams_for_topic(topic)
    for directed, expected in (
        (True, bigrams_to_dirgraph(bigrams)),
        (False, bigrams_to_undgraph(bigrams)),
    ):
        built = build_cooccurrence_graph(topic, directed=directed)
        assert built.is_directed() is directed
        assert nx.utils.edges_equal(built.edges(data="weight"), expected.edges(data="weight"))


def test_bigram_parity(topic, reference):
    bigrams = bigrams_for_topic(topic).sort_values(["type_1", "type_2"]).reset_index(drop=True)
    unordered = (
        bigrams_to_unordered(bigrams).sort_values(["type_1", "type_2"]).reset_index(drop=True)
    )

    assert_matches(bigrams, reference["bigrams"])
    assert_matches(unordered, reference["bigrams_unordered"])


@pytest.mark.parametrize("kind", ["directed", "undirected"])
def test_node_and_graph_stats_parity(topic, reference, kind):
    graph = build_cooccurrence_graph(topic, directed=kind == "directed")
    stats = node_stats(graph)

    assert_matches(stats.reset_index(drop=True), reference[f"node_stats_{kind}"])
    assert_matches(graph_stats(graph, stats), reference[f"graph_stats_{kind}"])


def test_node_stats_component_ids_order_by_size():
    graph = nx.Graph([("a", "b"), ("b", "c"), ("x", "y")])
    graph.add_node("solo")
    ids = node_stats(graph).set_index("node")["weak_component_id"].to_dict()

    assert ids == {"a": 1, "b": 1, "c": 1, "x": 2, "y": 2, "solo": 3}


def test_node_stats_directed_columns_and_strong_components():
    graph = nx.DiGraph([("a", "b"), ("b", "a"), ("b", "c")])
    out = node_stats(graph).set_index("node")

    assert list(out.reset_index().columns) == [*NODE_STATS_COLUMNS, *DIRECTED_NODE_STATS_COLUMNS]
    assert out["strong_component_id"].to_dict() == {"a": 1, "b": 1, "c": 2}
    assert out["weak_component_id"].nunique() == 1
    assert out.loc["b", ["in_degree", "out_degree"]].tolist() == [1, 2]


@pytest.mark.parametrize("graph", [nx.Graph(), nx.DiGraph()], ids=["undirected", "directed"])
def test_node_stats_of_empty_graph_has_full_schema(graph):
    expected = list(NODE_STATS_COLUMNS) + (
        list(DIRECTED_NODE_STATS_COLUMNS) if graph.is_directed() else []
    )
    assert list(node_stats(graph).columns) == expected


def test_graph_stats_counts_each_edge_once():
    graph = nx.Graph()
    graph.add_weighted_edges_from([("a", "b", 1), ("b", "c", 2), ("c", "a", 3), ("c", "d", 6)])
    stats = graph_stats(graph, node_stats(graph))

    assert stats["avg_degree"] == pytest.approx(graph.number_of_edges() / graph.number_of_nodes())
    assert stats["avg_strength"] == pytest.approx(12 / 4)
    assert stats["diameter"] == 2
    assert stats["components"] == 1


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_graph_stats_directed_uses_largest_strong_component():
    graph = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])
    graph.add_node("e")
    stats = graph_stats(graph, node_stats(graph))

    assert stats["diameter"] == 2
    assert stats["avg_path_length"] == pytest.approx(1.5)
    assert stats["components"] == 2


def test_graph_stats_of_empty_graph():
    stats = graph_stats(nx.Graph(), node_stats(nx.Graph()))

    assert stats["components"] == 0
    assert stats["assortativity"] is None


def test_leiden_parity_and_determinism(topic, reference):
    graph = build_cooccurrence_graph(topic, directed=False)
    communities = community_leiden(graph)

    assert [len(c) for c in communities] == sorted((len(c) for c in communities), reverse=True)
    assert partition_to_map(communities) == reference["leiden_undirected"]
    assert community_leiden(graph.to_directed()) == communities


def test_leiden_separates_disconnected_cliques():
    graph = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(4))

    assert sorted(map(sorted, community_leiden(graph))) == [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
    assert community_leiden(nx.Graph()) == []


def test_small_world_parity_and_progress(topic, reference):
    graph = build_cooccurrence_graph(topic, directed=False)
    calls = []
    result = small_world_indices(graph, n=3, progress_cb=lambda i, n: calls.append((i, n)))

    assert isinstance(result, SmallWorld)
    assert_matches(vars(result), reference["small_world_largest_component"])
    assert calls == [(0, 3), (1, 3), (2, 3), (3, 3)]


@pytest.mark.parametrize(
    "graph",
    [nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0)]), nx.Graph([(0, 0)]), nx.Graph([(0, 1), (2, 3)])],
    ids=["directed", "single-node", "disconnected"],
)
def test_small_world_rejects_invalid_graphs(graph):
    with pytest.raises(ValueError, match="small_world_indices requires"):
        small_world_indices(graph, n=1)


def test_component_helpers():
    graph = nx.DiGraph([("b", "c"), ("x", "y"), ("a", "z")])

    assert connected_components_sorted(graph) == [{"a", "z"}, {"b", "c"}, {"x", "y"}]
    assert not is_graph_connected(graph)
    assert is_graph_connected(nx.DiGraph([("a", "b"), ("c", "b")]))
    assert not is_graph_connected(nx.Graph())
    assert connected_components_sorted(nx.Graph()) == []
