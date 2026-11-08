"""Bigram co-occurrence graphs, node centralities, communities and graph-level metrics."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import igraph as ig
import leidenalg
import networkx as nx
import numpy as np
import pandas as pd

ProgressCallback = Callable[[int, int], None]
"""Called as ``callback(done, total)``."""

BIGRAM_COLUMNS = ("type_1", "type_2", "count")

NODE_STATS_COLUMNS = (
    "node",
    "degree",
    "degree_centrality",
    "strength",
    "betweenness",
    "closeness",
    "pagerank",
    "eigenvector",
    "clustering",
    "strong_component_id",
    "weak_component_id",
)
DIRECTED_NODE_STATS_COLUMNS = ("in_degree", "out_degree", "in_strength", "out_strength")


def _ordered_components(components: Iterable[set[Hashable]]) -> list[set[Hashable]]:
    return sorted(components, key=lambda comp: (-len(comp), min(str(n) for n in comp)))


def bigrams_for_topic(df: pd.DataFrame) -> pd.DataFrame:
    r"""Count consecutive type pairs within each informant's response list.

    Parameters
    ----------
    df : pandas.DataFrame
        Topic frame with columns ``informant_id``, ``pos`` and ``type``.

    Returns
    -------
    pandas.DataFrame
        Columns ``type_1``, ``type_2`` and ``count``, one row per ordered pair
        ``(type at pos i, type at pos i + 1)`` observed for the same informant.

    Notes
    -----
    Runs in :math:`O(R \log R)` for :math:`R` responses.
    """
    if df.empty:
        return pd.DataFrame(columns=list(BIGRAM_COLUMNS))

    df = df[["informant_id", "pos", "type"]].sort_values(["informant_id", "pos"])
    df = df.assign(type_2=df.groupby("informant_id")["type"].shift(-1)).dropna(subset=["type_2"])
    if df.empty:
        return pd.DataFrame(columns=list(BIGRAM_COLUMNS))

    return (
        df.groupby(["type", "type_2"])
        .size()
        .reset_index(name="count")
        .rename(columns={"type": "type_1"})
    )


def bigrams_to_unordered(df_bigrams: pd.DataFrame) -> pd.DataFrame:
    """Merge ``(a, b)`` and ``(b, a)`` into one unordered pair, summing counts.

    Parameters
    ----------
    df_bigrams : pandas.DataFrame
        Columns ``type_1``, ``type_2`` and ``count``.

    Returns
    -------
    pandas.DataFrame
        Same columns, with ``type_1 <= type_2`` compared as strings.
    """
    if df_bigrams.empty:
        return pd.DataFrame(columns=list(BIGRAM_COLUMNS))

    first = df_bigrams["type_1"].astype(str).to_numpy()
    second = df_bigrams["type_2"].astype(str).to_numpy()
    swap = first > second
    out = pd.DataFrame(
        {
            "type_1": np.where(swap, second, first),
            "type_2": np.where(swap, first, second),
            "count": df_bigrams["count"].to_numpy(),
        }
    )
    return out.groupby(["type_1", "type_2"], dropna=False)["count"].sum().reset_index()


def _edgelist_to_graph(df_bigrams: pd.DataFrame, create_using: type[nx.Graph]) -> nx.Graph:
    return nx.from_pandas_edgelist(
        df_bigrams.rename(columns={"count": "weight"}),
        source="type_1",
        target="type_2",
        edge_attr="weight",
        create_using=create_using,
    )


def bigrams_to_dirgraph(df_bigrams: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph whose edge ``weight`` is the bigram count.

    Parameters
    ----------
    df_bigrams : pandas.DataFrame
        Columns ``type_1``, ``type_2`` and ``count``.

    Returns
    -------
    networkx.DiGraph
    """
    return _edgelist_to_graph(df_bigrams, nx.DiGraph)


def bigrams_to_undgraph(df_bigrams: pd.DataFrame) -> nx.Graph:
    """Build an undirected graph whose edge ``weight`` is the unordered bigram count.

    Parameters
    ----------
    df_bigrams : pandas.DataFrame
        Columns ``type_1``, ``type_2`` and ``count``.

    Returns
    -------
    networkx.Graph
    """
    return _edgelist_to_graph(bigrams_to_unordered(df_bigrams), nx.Graph)


def build_cooccurrence_graph(df: pd.DataFrame, *, directed: bool) -> nx.Graph:
    """Build the weighted bigram co-occurrence graph of a topic frame.

    Parameters
    ----------
    df : pandas.DataFrame
        Topic frame with columns ``informant_id``, ``pos`` and ``type``.
    directed : bool
        Return a :class:`networkx.DiGraph` when true, otherwise an undirected
        graph with ``(a, b)`` and ``(b, a)`` merged.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
    """
    bigrams = bigrams_for_topic(df)
    return bigrams_to_dirgraph(bigrams) if directed else bigrams_to_undgraph(bigrams)


def node_stats(graph: nx.Graph) -> pd.DataFrame:
    """Compute per-node degree, centrality, clustering and component membership.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Graph with an optional ``weight`` edge attribute.

    Returns
    -------
    pandas.DataFrame
        Columns listed in ``NODE_STATS_COLUMNS`` (plus
        ``DIRECTED_NODE_STATS_COLUMNS`` for directed graphs), sorted by
        ``strength`` and ``degree`` descending, then ``node`` ascending:

        * ``degree``: unweighted degree; ``degree_centrality``: degree / (V - 1);
        * ``strength``: weighted degree;
        * ``betweenness``: unnormalized betweenness centrality;
        * ``closeness``: closeness with the Wasserman-Faust correction;
        * ``pagerank``: weighted PageRank, damping 0.85, tolerance 1e-6;
        * ``eigenvector``: weighted eigenvector centrality (``max_iter=1000``);
        * ``clustering``: weighted clustering coefficient;
        * ``strong_component_id`` / ``weak_component_id``: 1-based component
          index, components ordered by size descending.

    Notes
    -----
    Betweenness dominates the cost at :math:`O(VE)` (Brandes, 2001).
    """
    directed = graph.is_directed()
    columns = list(NODE_STATS_COLUMNS) + (list(DIRECTED_NODE_STATS_COLUMNS) if directed else [])
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=columns)

    if directed:
        strong = _ordered_components(nx.strongly_connected_components(graph))
        weak = _ordered_components(nx.weakly_connected_components(graph))
    else:
        strong = weak = _ordered_components(nx.connected_components(graph))

    metrics: dict[str, Mapping[Hashable, Any]] = {
        "degree": dict(graph.degree()),
        "degree_centrality": nx.degree_centrality(graph),
        "strength": dict(graph.degree(weight="weight")),
        "betweenness": nx.betweenness_centrality(graph, normalized=False),
        "closeness": nx.closeness_centrality(graph, wf_improved=True),
        "pagerank": nx.pagerank(graph, weight="weight", alpha=0.85, tol=1e-6),
        "eigenvector": nx.eigenvector_centrality(graph, weight="weight", max_iter=1000),
        "clustering": nx.clustering(graph, weight="weight"),
        "strong_component_id": partition_to_map(strong),
        "weak_component_id": partition_to_map(weak),
    }
    if directed:
        metrics |= {
            "in_degree": dict(graph.in_degree()),
            "out_degree": dict(graph.out_degree()),
            "in_strength": dict(graph.in_degree(weight="weight")),
            "out_strength": dict(graph.out_degree(weight="weight")),
        }

    nodes = list(graph.nodes())
    out = pd.DataFrame(
        {"node": nodes} | {name: [m[n] for n in nodes] for name, m in metrics.items()}
    )
    return out[columns].sort_values(["strength", "degree", "node"], ascending=[False, False, True])


def community_leiden(graph: nx.Graph, seed: int = 42) -> list[set[Hashable]]:
    """Detect communities with the Leiden algorithm on the weighted graph.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Directed graphs are converted to undirected first.
    seed : int, default 42
        Seed passed to :func:`leidenalg.find_partition`.

    Returns
    -------
    list of set
        One node set per community, ordered by size descending.

    Notes
    -----
    Optimizes modularity (``ModularityVertexPartition``) as in Traag, Waltman
    and van Eck (2019).
    """
    undirected = graph.to_undirected() if graph.is_directed() else graph
    if undirected.number_of_nodes() == 0:
        return []

    nodes = list(undirected.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    edges = list(undirected.edges(data="weight", default=1.0))
    ig_graph = ig.Graph(n=len(nodes), edges=[(index[u], index[v]) for u, v, _ in edges])
    ig_graph.es["weight"] = [w for _, _, w in edges]

    partition = leidenalg.find_partition(
        ig_graph, leidenalg.ModularityVertexPartition, weights="weight", seed=seed
    )
    communities = [{nodes[i] for i in community} for community in partition]
    return sorted(communities, key=len, reverse=True)


def partition_to_map(communities: Iterable[Iterable[Hashable]]) -> dict[Hashable, int]:
    """Map each node to the 1-based index of the community that contains it.

    Parameters
    ----------
    communities : iterable of iterable
        Disjoint node collections, e.g. the output of :func:`community_leiden`.

    Returns
    -------
    dict
        ``node -> community index``, numbered in iteration order from 1.
    """
    return {node: i for i, community in enumerate(communities, start=1) for node in community}


@dataclass(frozen=True)
class SmallWorld:
    """Small-world indices of a connected undirected graph.

    Attributes
    ----------
    swi, omega : float or None
        Index value, ``None`` when undefined.
    swi_error, omega_error : str or None
        Reason code when the index is undefined: ``"zero_path_length"``,
        ``"zero_lattice_clustering"``, ``"zero_lattice_path_length"``,
        ``"zero_path_length_range"`` or ``"zero_clustering_range"``.
    """

    swi: float | None
    omega: float | None
    swi_error: str | None
    omega_error: str | None


def small_world_indices(
    graph: nx.Graph, *, n: int = 10, progress_cb: ProgressCallback | None = None
) -> SmallWorld:
    r"""Compute the small-world propensity indices ω and SWI.

    Parameters
    ----------
    graph : networkx.Graph
        Connected, undirected graph with at least four nodes.
    n : int, default 10
        Number of lattice and random reference graphs. Reference ``i`` uses
        seed ``42 + i``.
    progress_cb : callable, optional
        Called with ``(i, n)`` before each reference pair and ``(n, n)`` at the end.

    Returns
    -------
    SmallWorld

    Raises
    ------
    ValueError
        If the graph is directed, has fewer than two nodes or is disconnected.

    Notes
    -----
    With :math:`L, C` the average shortest path length and average clustering
    of ``graph``, :math:`L_r, C_r` their means over the random references,
    :math:`L_l` the mean path length over the lattice references and
    :math:`C_l` the maximum lattice clustering,

    .. math::

        \omega = \frac{L_r}{L} - \frac{C}{C_l}, \qquad
        \mathrm{SWI} = \frac{L - L_l}{L_r - L_l} \cdot \frac{C - C_r}{C_l - C_r}

    (Telesford et al., 2011; Neal, 2017). References are built with
    ``lattice_reference(niter=5)`` and ``random_reference(niter=10)``; each
    costs :math:`O(\mathrm{niter} \cdot E)` swaps plus an all-pairs shortest
    path computation.
    """
    if graph.is_directed():
        msg = "small_world_indices requires an undirected graph"
        raise ValueError(msg)
    if graph.number_of_nodes() < 2:
        msg = "small_world_indices requires at least two nodes"
        raise ValueError(msg)
    if not nx.is_connected(graph):
        msg = "small_world_indices requires a connected graph"
        raise ValueError(msg)

    L = nx.average_shortest_path_length(graph)
    C = nx.average_clustering(graph)

    lattice_clustering: list[float] = []
    lattice_paths: list[float] = []
    random_clustering: list[float] = []
    random_paths: list[float] = []
    for i in range(n):
        if progress_cb:
            progress_cb(i, n)
        lattice = nx.algorithms.smallworld.lattice_reference(graph, niter=5, seed=42 + i)
        random = nx.algorithms.smallworld.random_reference(graph, niter=10, seed=42 + i)
        lattice_clustering.append(nx.average_clustering(lattice))
        random_clustering.append(nx.average_clustering(random))
        lattice_paths.append(nx.average_shortest_path_length(lattice))
        random_paths.append(nx.average_shortest_path_length(random))
    if progress_cb:
        progress_cb(n, n)

    Cl = max(lattice_clustering, default=-1.0)
    Cr = sum(random_clustering) / n if n else 0.0
    Ll = sum(lattice_paths) / n if n else 0.0
    Lr = sum(random_paths) / n if n else 0.0

    omega: float | None = None
    omega_error: str | None = None
    if L == 0:
        omega_error = "zero_path_length"
    elif Cl <= 0:
        omega_error = "zero_lattice_clustering"
    else:
        omega = (Lr / L) - (C / Cl)

    swi: float | None = None
    swi_error: str | None = None
    if Ll == 0:
        swi_error = "zero_lattice_path_length"
    elif Lr - Ll == 0:
        swi_error = "zero_path_length_range"
    elif Cl - Cr == 0:
        swi_error = "zero_clustering_range"
    else:
        swi = ((L - Ll) / (Lr - Ll)) * ((C - Cr) / (Cl - Cr))

    return SmallWorld(swi=swi, omega=omega, swi_error=swi_error, omega_error=omega_error)


def is_graph_connected(graph: nx.Graph) -> bool:
    """Return whether the graph is connected (weakly, for directed graphs).

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph

    Returns
    -------
    bool
        ``False`` for the empty graph.
    """
    if graph.number_of_nodes() == 0:
        return False
    if graph.is_directed():
        return bool(nx.is_weakly_connected(graph))
    return bool(nx.is_connected(graph))


def connected_components_sorted(graph: nx.Graph) -> list[set[Hashable]]:
    """Return the (weakly) connected components, largest first.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph

    Returns
    -------
    list of set
        Ties in size are broken by the smallest node label as a string.
    """
    if graph.number_of_nodes() == 0:
        return []
    if graph.is_directed():
        return _ordered_components(nx.weakly_connected_components(graph))
    return _ordered_components(nx.connected_components(graph))


def graph_stats(graph: nx.Graph, node_stats_df: pd.DataFrame) -> dict[str, float | int | None]:
    """Compute graph-level structural metrics.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
    node_stats_df : pandas.DataFrame
        Must contain ``degree`` and ``strength``; the output of
        :func:`node_stats` qualifies.

    Returns
    -------
    dict
        ``diameter`` and ``avg_path_length`` of the largest strongly
        connected (directed) or connected (undirected) component;
        ``avg_clustering`` (weighted); ``avg_degree`` and ``avg_strength``;
        ``density``; ``components`` (weakly connected components for directed
        graphs); ``assortativity`` (degree assortativity, ``None`` if undefined).

    Notes
    -----
    ``avg_degree`` and ``avg_strength`` are half the mean node degree and
    strength, i.e. :math:`E/V` and :math:`W/V` for an undirected graph with
    total weight :math:`W`, so that each edge is counted once.
    """
    if graph.number_of_nodes() == 0:
        return {
            "diameter": 0,
            "avg_clustering": 0.0,
            "avg_degree": 0.0,
            "avg_strength": 0.0,
            "density": 0.0,
            "components": 0,
            "avg_path_length": 0.0,
            "assortativity": None,
        }

    if graph.is_directed():
        cores = list(nx.strongly_connected_components(graph))
        n_components = nx.number_weakly_connected_components(graph)
    else:
        cores = list(nx.connected_components(graph))
        n_components = len(cores)
    largest = graph.subgraph(max(cores, key=len))

    if largest.number_of_nodes() <= 1:
        diameter, avg_path_length = 0, 0.0
    else:
        diameter = nx.diameter(largest)
        avg_path_length = nx.average_shortest_path_length(largest)

    try:
        assortativity: float | None = float(nx.degree_assortativity_coefficient(graph))
    except (nx.NetworkXError, ZeroDivisionError):
        assortativity = None

    return {
        "diameter": diameter,
        "avg_clustering": float(nx.average_clustering(graph, weight="weight")),
        "avg_degree": float(node_stats_df["degree"].mean() / 2)
        if "degree" in node_stats_df
        else 0.0,
        "avg_strength": (
            float(node_stats_df["strength"].mean() / 2) if "strength" in node_stats_df else 0.0
        ),
        "density": float(nx.density(graph)),
        "components": n_components,
        "avg_path_length": avg_path_length,
        "assortativity": assortativity,
    }
