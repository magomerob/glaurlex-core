"""Structural comparison of co-occurrence graphs.

Every function takes graphs keyed by an arbitrary label (a topic, an informant
group, or a combination); the meaning of the label is left to the caller.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from glaurlex.core.graph import ProgressCallback, community_leiden, graph_stats

BuildGraph = Callable[[Sequence[Hashable]], nx.Graph]
"""Builds a graph from a subset of informant ids."""

Edge = tuple[Hashable, Hashable]

ONLY_A = "only_a"
ONLY_B = "only_b"
BOTH = "both"

TWO = 2
MIN_COMMON_NODES = 2
MIN_SHARED_FOR_CORR = 3
MIN_N_FOR_RANK_CORR = 3
MIN_UNIQUE_FOR_CORR = 2

PROFILE_METRICS = (
    "n_nodes",
    "n_edges",
    "density",
    "avg_degree",
    "avg_strength",
    "avg_clustering",
    "diameter",
    "avg_path_length",
    "components",
    "largest_component_fraction",
    "assortativity",
    "n_communities",
    "modularity",
)
"""Keys returned by :func:`graph_profile`, in display order."""

COMMUNITY_METRICS = ("n_communities", "modularity")
RAREFACTION_METRICS = tuple(k for k in PROFILE_METRICS if k not in COMMUNITY_METRICS)


def canonical_edges(graph: nx.Graph) -> dict[Edge, float]:
    """Return every edge with its weight under a canonical key.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Missing ``weight`` attributes count as 1.

    Returns
    -------
    dict
        ``(u, v) -> weight``. Undirected edges are keyed by their endpoints as
        sorted strings, so ``(a, b)`` and ``(b, a)`` coincide.
    """
    directed = graph.is_directed()
    out: dict[Edge, float] = {}
    for u, v, w in graph.edges(data="weight", default=1.0):
        if directed:
            edge: Edge = (u, v)
        else:
            first, second = sorted((str(u), str(v)))
            edge = (first, second)
        out[edge] = out.get(edge, 0.0) + float(w)
    return out


def _normalized(weights: Mapping[Edge, float]) -> dict[Edge, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        return dict.fromkeys(weights, 0.0)
    return {k: float(v) / total for k, v in weights.items()}


def _set_metrics(set_a: set[Any], set_b: set[Any]) -> dict[str, float | int]:
    common = len(set_a & set_b)
    union = len(set_a | set_b)
    smallest = min(len(set_a), len(set_b))
    return {
        "n_a": len(set_a),
        "n_b": len(set_b),
        "n_common": common,
        "n_union": union,
        "jaccard": common / union if union else 0.0,
        "overlap_coef": common / smallest if smallest else 0.0,
        "dice": 2 * common / (len(set_a) + len(set_b)) if union else 0.0,
    }


def compare_pair_nodes(graph_a: nx.Graph, graph_b: nx.Graph) -> dict[str, Any]:
    r"""Compare the node sets of two graphs.

    Parameters
    ----------
    graph_a, graph_b : networkx.Graph

    Returns
    -------
    dict
        ``n_a``, ``n_b``, ``n_common``, ``n_union``, ``jaccard``
        (:math:`|A \cap B| / |A \cup B|`), ``overlap_coef``
        (:math:`|A \cap B| / \min(|A|, |B|)`), ``dice``
        (:math:`2|A \cap B| / (|A| + |B|)`), and the sorted node lists
        ``only_a``, ``only_b`` and ``common``.
    """
    set_a, set_b = set(graph_a.nodes()), set(graph_b.nodes())
    return _set_metrics(set_a, set_b) | {
        "only_a": sorted(set_a - set_b, key=str),
        "only_b": sorted(set_b - set_a, key=str),
        "common": sorted(set_a & set_b, key=str),
    }


def compare_pair_edges(graph_a: nx.Graph, graph_b: nx.Graph) -> dict[str, Any]:
    """Compare the edge sets of two graphs and the weights of their shared edges.

    Parameters
    ----------
    graph_a, graph_b : networkx.Graph

    Returns
    -------
    dict
        The set indices of :func:`compare_pair_nodes` over canonical edges,
        the sorted edge lists ``only_a``, ``only_b`` and ``common``,
        ``weight_spearman`` / ``weight_spearman_p`` (Spearman correlation of
        the shared edges' normalized weights), ``weight_cosine`` (cosine
        similarity of the same vectors), and ``norm_weights_a`` /
        ``norm_weights_b``. Correlations are NaN with fewer than 3 shared edges.

    Notes
    -----
    Weights are divided by each graph's total weight, so the correlation
    reflects relative structure rather than the number of informants behind
    each graph.
    """
    edges_a, edges_b = canonical_edges(graph_a), canonical_edges(graph_b)
    set_a, set_b = set(edges_a), set(edges_b)
    common = sorted(set_a & set_b, key=str)
    out: dict[str, Any] = _set_metrics(set_a, set_b) | {
        "only_a": sorted(set_a - set_b, key=str),
        "only_b": sorted(set_b - set_a, key=str),
        "common": common,
    }

    norm_a, norm_b = _normalized(edges_a), _normalized(edges_b)
    if len(common) >= MIN_SHARED_FOR_CORR:
        va = np.array([norm_a[e] for e in common])
        vb = np.array([norm_b[e] for e in common])
        rho, p_rho = stats.spearmanr(va, vb)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        out["weight_spearman"] = float(rho)
        out["weight_spearman_p"] = float(p_rho)
        out["weight_cosine"] = float(va @ vb / denom) if denom > 0 else float("nan")
    else:
        out["weight_spearman"] = out["weight_spearman_p"] = out["weight_cosine"] = float("nan")

    out["norm_weights_a"] = norm_a
    out["norm_weights_b"] = norm_b
    return out


def edge_weight_divergence(graph_a: nx.Graph, graph_b: nx.Graph) -> pd.DataFrame:
    """Tabulate every edge of either graph by normalized-weight difference.

    Parameters
    ----------
    graph_a, graph_b : networkx.Graph

    Returns
    -------
    pandas.DataFrame
        Columns ``edge`` (``"u — v"``), ``source``, ``target``,
        ``norm_weight_a``, ``norm_weight_b`` (0 where the edge is absent),
        ``delta`` (B minus A) and ``presence`` (``ONLY_A``, ``ONLY_B`` or
        ``BOTH``), sorted by ``|delta|`` descending.
    """
    columns = ["edge", "source", "target", "norm_weight_a", "norm_weight_b", "delta", "presence"]
    norm_a = _normalized(canonical_edges(graph_a))
    norm_b = _normalized(canonical_edges(graph_b))
    edges = sorted(norm_a.keys() | norm_b.keys(), key=str)
    if not edges:
        return pd.DataFrame(columns=columns)

    in_a = np.array([e in norm_a for e in edges])
    in_b = np.array([e in norm_b for e in edges])
    weight_a = np.array([norm_a.get(e, 0.0) for e in edges])
    weight_b = np.array([norm_b.get(e, 0.0) for e in edges])
    out = pd.DataFrame(
        {
            "edge": [f"{u} — {v}" for u, v in edges],
            "source": [u for u, _ in edges],
            "target": [v for _, v in edges],
            "norm_weight_a": weight_a,
            "norm_weight_b": weight_b,
            "delta": weight_b - weight_a,
            "presence": np.where(in_a & in_b, BOTH, np.where(in_a, ONLY_A, ONLY_B)),
        }
    )
    return out.reindex(out["delta"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def _presence(graphs: Mapping[str, nx.Graph]) -> pd.DataFrame:
    nodes = sorted(set().union(*(g.nodes() for g in graphs.values())), key=str)
    return pd.DataFrame(
        {label: pd.Index(nodes).isin(list(g.nodes())) for label, g in graphs.items()},
        index=pd.Index(nodes, dtype=object),
    )


def node_presence_matrix(graphs: Mapping[str, nx.Graph]) -> pd.DataFrame:
    """Tabulate which graphs contain each node.

    Parameters
    ----------
    graphs : mapping of str to networkx.Graph

    Returns
    -------
    pandas.DataFrame
        Columns ``node``, one boolean column per label, ``n_graphs`` and
        ``exclusive_to`` (the only label containing the node, else ``None``),
        sorted by ``n_graphs`` descending then ``node``.
    """
    labels = list(graphs)
    presence = _presence(graphs)
    if presence.empty:
        return pd.DataFrame(columns=["node", *labels, "n_graphs", "exclusive_to"])

    n_graphs = presence.sum(axis=1)
    exclusive = presence.idxmax(axis=1).where(n_graphs == 1, None)
    out = (
        presence.assign(n_graphs=n_graphs, exclusive_to=exclusive).rename_axis("node").reset_index()
    )
    return out.sort_values(["n_graphs", "node"], ascending=[False, True]).reset_index(drop=True)


def set_summary(graphs: Mapping[str, nx.Graph]) -> pd.DataFrame:
    """Summarize each graph's node set relative to the others.

    Parameters
    ----------
    graphs : mapping of str to networkx.Graph

    Returns
    -------
    pandas.DataFrame
        Columns ``entity``, ``nodes``, ``edges``, ``exclusive`` (nodes in no
        other graph), ``pct_exclusive`` and ``in_all`` (nodes shared by every
        graph).
    """
    columns = ["entity", "nodes", "edges", "exclusive", "pct_exclusive", "in_all"]
    presence = _presence(graphs)
    if not graphs:
        return pd.DataFrame(columns=columns)

    n_graphs = presence.sum(axis=1)
    exclusive = presence[n_graphs == 1].sum(axis=0)
    nodes = presence.sum(axis=0)
    return pd.DataFrame(
        {
            "entity": list(graphs),
            "nodes": [int(nodes[label]) for label in graphs],
            "edges": [g.number_of_edges() for g in graphs.values()],
            "exclusive": [int(exclusive[label]) for label in graphs],
            "pct_exclusive": [
                100.0 * exclusive[label] / nodes[label] if nodes[label] else 0.0 for label in graphs
            ],
            "in_all": int((n_graphs == len(graphs)).sum()),
        },
        columns=columns,
    )


def pairwise_similarity_matrix(graphs: Mapping[str, nx.Graph], on: str = "nodes") -> pd.DataFrame:
    """Compute the Jaccard index between every pair of graphs.

    Parameters
    ----------
    graphs : mapping of str to networkx.Graph
    on : {"nodes", "edges"}, default "nodes"
        Compare node sets or canonical edge sets.

    Returns
    -------
    pandas.DataFrame
        Symmetric square matrix indexed by label, 1 on the diagonal.

    Raises
    ------
    ValueError
        If ``on`` is neither ``"nodes"`` nor ``"edges"``.
    """
    if on not in {"nodes", "edges"}:
        msg = "on must be 'nodes' or 'edges'"
        raise ValueError(msg)
    labels = list(graphs)
    sets = {
        label: set(g.nodes()) if on == "nodes" else set(canonical_edges(g))
        for label, g in graphs.items()
    }
    matrix = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            union = len(sets[a] | sets[b])
            matrix.loc[a, b] = matrix.loc[b, a] = len(sets[a] & sets[b]) / union if union else 0.0
    return matrix


def pairwise_similarity_table(graphs: Mapping[str, nx.Graph]) -> pd.DataFrame:
    """List node and edge Jaccard indices for every unordered pair of graphs.

    Parameters
    ----------
    graphs : mapping of str to networkx.Graph

    Returns
    -------
    pandas.DataFrame
        Columns ``entity_a``, ``entity_b``, ``jaccard_nodes`` and
        ``jaccard_edges``, sorted by ``jaccard_nodes`` descending.
    """
    nodes = pairwise_similarity_matrix(graphs, on="nodes")
    edges = pairwise_similarity_matrix(graphs, on="edges")
    labels = list(graphs)
    pairs = [(a, b) for i, a in enumerate(labels) for b in labels[i + 1 :]]
    out = pd.DataFrame(
        {
            "entity_a": [a for a, _ in pairs],
            "entity_b": [b for _, b in pairs],
            "jaccard_nodes": [float(nodes.loc[a, b]) for a, b in pairs],
            "jaccard_edges": [float(edges.loc[a, b]) for a, b in pairs],
        }
    )
    return out.sort_values("jaccard_nodes", ascending=False).reset_index(drop=True)


def intersection_sizes(graphs: Mapping[str, nx.Graph], top: int = 15) -> pd.DataFrame:
    """Count nodes by the exact combination of graphs that contain them.

    Parameters
    ----------
    graphs : mapping of str to networkx.Graph
    top : int, default 15
        Number of largest combinations returned.

    Returns
    -------
    pandas.DataFrame
        Columns ``combination`` (labels joined by ``" ∩ "``), ``n_entities``
        and ``n_nodes``, sorted by ``n_nodes`` descending then ``combination``.
    """
    columns = ["combination", "n_entities", "n_nodes"]
    presence = _presence(graphs)
    if presence.empty:
        return pd.DataFrame(columns=columns)

    labels = np.array(list(graphs), dtype=object)
    out = (
        pd.DataFrame(
            {
                "combination": [" ∩ ".join(labels[row]) for row in presence.to_numpy()],
                "n_entities": presence.sum(axis=1).to_numpy(),
            }
        )
        .groupby(["combination", "n_entities"])
        .size()
        .reset_index(name="n_nodes")
    )
    return (
        out.sort_values(["n_nodes", "combination"], ascending=[False, True])
        .reset_index(drop=True)
        .head(top)
    )


def _largest_component_fraction(graph: nx.Graph) -> float:
    n_nodes = graph.number_of_nodes()
    if n_nodes == 0:
        return 0.0
    undirected = graph.to_undirected() if graph.is_directed() else graph
    return float(max(len(c) for c in nx.connected_components(undirected)) / n_nodes)


def degree_frame(graph: nx.Graph) -> pd.DataFrame:
    """Return the ``node``, ``degree`` and ``strength`` columns of a graph.

    Parameters
    ----------
    graph : networkx.Graph

    Returns
    -------
    pandas.DataFrame
        Enough input for :func:`~glaurlex.core.graph.graph_stats` at
        :math:`O(V + E)`, without the centralities of
        :func:`~glaurlex.core.graph.node_stats`.
    """
    nodes = list(graph.nodes())
    degree = dict(graph.degree())
    strength = dict(graph.degree(weight="weight"))
    return pd.DataFrame(
        {
            "node": nodes,
            "degree": [degree[n] for n in nodes],
            "strength": [strength[n] for n in nodes],
        }
    )


def _base_profile(graph: nx.Graph, node_stats_df: pd.DataFrame | None = None) -> dict[str, Any]:
    base = graph_stats(graph, degree_frame(graph) if node_stats_df is None else node_stats_df)
    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "density": base["density"],
        "avg_degree": base["avg_degree"],
        "avg_strength": base["avg_strength"],
        "avg_clustering": base["avg_clustering"],
        "diameter": base["diameter"],
        "avg_path_length": base["avg_path_length"],
        "components": base["components"],
        "largest_component_fraction": _largest_component_fraction(graph),
        "assortativity": base["assortativity"],
    }


def graph_profile(
    graph: nx.Graph,
    node_stats_df: pd.DataFrame | None = None,
    community_map: Mapping[Hashable, Hashable] | None = None,
) -> dict[str, Any]:
    """Compute the structural profile used to compare graphs.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
    node_stats_df : pandas.DataFrame, optional
        Frame with ``degree`` and ``strength``; :func:`degree_frame` is used
        when omitted.
    community_map : mapping, optional
        ``node -> community``. Leiden communities (seed 42) are computed when
        omitted.

    Returns
    -------
    dict
        The keys in ``PROFILE_METRICS``: the metrics of
        :func:`~glaurlex.core.graph.graph_stats` plus ``n_nodes``,
        ``n_edges``, ``largest_component_fraction``, ``n_communities`` and
        ``modularity`` (weighted, on the undirected graph; NaN unless the
        partition covers every node).
    """
    profile = _base_profile(graph, node_stats_df)
    communities: list[set[Hashable]] = []
    modularity = float("nan")
    if graph.number_of_nodes():
        if community_map is None:
            communities = community_leiden(graph, seed=42)
        else:
            grouped: dict[Hashable, set[Hashable]] = {}
            for node, cid in community_map.items():
                grouped.setdefault(cid, set()).add(node)
            communities = sorted(grouped.values(), key=len, reverse=True)

        undirected = graph.to_undirected() if graph.is_directed() else graph
        if communities and set().union(*communities) == set(graph.nodes()):
            try:
                modularity = float(
                    nx.community.modularity(undirected, communities, weight="weight")
                )
            except (nx.NetworkXError, ZeroDivisionError, ValueError):
                modularity = float("nan")

    return profile | {"n_communities": len(communities), "modularity": modularity}


def profile_table(profiles: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Lay out several structural profiles side by side.

    Parameters
    ----------
    profiles : mapping of str to dict
        ``label -> graph_profile(...)``.

    Returns
    -------
    pandas.DataFrame
        A ``metric`` column with the keys of ``PROFILE_METRICS`` and one
        column per label. With exactly two labels ``a, b``, also ``delta``
        (``b - a``) and ``delta_pct`` (``100 * delta / |a|``, NaN when ``a`` is 0).
    """
    labels = list(profiles)
    out = pd.DataFrame(
        {
            "metric": list(PROFILE_METRICS),
            **{
                label: [
                    float("nan") if profiles[label].get(k) is None else float(profiles[label][k])
                    for k in PROFILE_METRICS
                ]
                for label in labels
            },
        }
    )
    if len(labels) == TWO:
        a, b = out[labels[0]], out[labels[1]]
        out["delta"] = b - a
        with np.errstate(divide="ignore", invalid="ignore"):
            out["delta_pct"] = np.where(a.abs() > 0, 100.0 * (b - a) / a.abs(), np.nan)
    return out


def degree_ccdf(graph: nx.Graph) -> pd.DataFrame:
    r"""Compute the complementary cumulative degree distribution.

    Parameters
    ----------
    graph : networkx.Graph

    Returns
    -------
    pandas.DataFrame
        Columns ``degree`` (each distinct degree :math:`k`) and ``ccdf``
        (:math:`P(\deg \ge k)`).
    """
    degrees = np.sort(np.fromiter((d for _, d in graph.degree()), dtype=float))
    if degrees.size == 0:
        return pd.DataFrame(columns=["degree", "ccdf"])
    unique = np.unique(degrees)
    ccdf = (degrees.size - np.searchsorted(degrees, unique, side="left")) / degrees.size
    return pd.DataFrame({"degree": unique, "ccdf": ccdf})


def node_metric_map(node_stats_df: pd.DataFrame, metric: str) -> dict[Hashable, float]:
    """Extract ``node -> value`` for one column of a node statistics frame.

    Parameters
    ----------
    node_stats_df : pandas.DataFrame
        Output of :func:`~glaurlex.core.graph.node_stats`.
    metric : str

    Returns
    -------
    dict
        Empty when the frame is empty or lacks ``metric``.
    """
    if node_stats_df.empty or metric not in node_stats_df.columns:
        return {}
    return dict(zip(node_stats_df["node"], node_stats_df[metric].astype(float), strict=True))


def node_metric_table(node_stats_by_label: Mapping[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    """Compare one node metric across graphs, with values and ranks.

    Parameters
    ----------
    node_stats_by_label : mapping of str to pandas.DataFrame
        ``label -> node_stats(graph)``.
    metric : str
        Column of the node statistics, e.g. ``"pagerank"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``node``, ``<label>`` and ``<label> (rank)`` per label (rank 1
        is the highest value, ties take the minimum rank). Nodes absent from a
        graph are NaN there, not 0. With two labels ``a, b``, also
        ``delta_value`` (``b - a``), ``delta_rank`` (``rank_a - rank_b``) and
        ``presence``, sorted by ``delta_value`` descending; otherwise sorted
        by the first label's value. Ties keep ``node`` order.

    Raises
    ------
    KeyError
        If a non-empty frame lacks ``metric``.
    """
    labels = list(node_stats_by_label)
    frames = []
    for label in labels:
        df = node_stats_by_label[label]
        rank_col = f"{label} (rank)"
        if df.empty:
            frames.append(
                pd.DataFrame({"node": [], label: [], rank_col: pd.Series([], dtype="Int64")})
            )
            continue
        if metric not in df.columns:
            msg = f"node statistics for {label!r} have no column {metric!r}"
            raise KeyError(msg)
        sub = df[["node", metric]].rename(columns={metric: label})
        frames.append(
            sub.assign(**{rank_col: sub[label].rank(ascending=False, method="min").astype("Int64")})
        )

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="node", how="outer")
    if out.empty:
        return out

    if len(labels) == TWO:
        a, b = labels
        out["delta_value"] = out[b] - out[a]
        out["delta_rank"] = out[f"{a} (rank)"] - out[f"{b} (rank)"]
        out["presence"] = np.where(
            out[a].notna() & out[b].notna(), BOTH, np.where(out[a].notna(), ONLY_A, ONLY_B)
        )
        key = "delta_value"
    else:
        key = labels[0]
    out = out.sort_values("node", key=lambda s: s.astype(str), kind="stable").sort_values(
        key, ascending=False, na_position="last", kind="stable"
    )
    return out.reset_index(drop=True)


def rank_correlation(
    values_a: Mapping[Hashable, float], values_b: Mapping[Hashable, float]
) -> dict[str, Any]:
    """Correlate two node rankings over the nodes they share.

    Parameters
    ----------
    values_a, values_b : mapping
        ``node -> value``, e.g. from :func:`node_metric_map`.

    Returns
    -------
    dict
        ``n``, ``spearman_rho``, ``spearman_p``, ``kendall_tau_b`` and
        ``kendall_p``; or ``n`` and ``error`` when fewer than 3 shared
        non-NaN values remain or either side is constant.
    """
    common = sorted(set(values_a) & set(values_b), key=str)
    va = np.array([values_a[n] for n in common], dtype=float)
    vb = np.array([values_b[n] for n in common], dtype=float)
    keep = ~(np.isnan(va) | np.isnan(vb))
    va, vb = va[keep], vb[keep]
    n = int(va.size)
    if (
        n < MIN_N_FOR_RANK_CORR
        or np.unique(va).size < MIN_UNIQUE_FOR_CORR
        or np.unique(vb).size < MIN_UNIQUE_FOR_CORR
    ):
        return {"n": n, "error": "insufficient_data"}

    rho, p_rho = stats.spearmanr(va, vb)
    tau, p_tau = stats.kendalltau(va, vb, variant="b")
    return {
        "n": n,
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "kendall_tau_b": float(tau),
        "kendall_p": float(p_tau),
    }


def _contingency(
    part_a: Mapping[Hashable, Hashable],
    part_b: Mapping[Hashable, Hashable],
    nodes: Sequence[Hashable],
) -> np.ndarray:
    codes_a, uniques_a = pd.factorize(pd.Series([part_a[n] for n in nodes], dtype=object))
    codes_b, uniques_b = pd.factorize(pd.Series([part_b[n] for n in nodes], dtype=object))
    table = np.zeros((len(uniques_a), len(uniques_b)))
    np.add.at(table, (codes_a, codes_b), 1)
    return table


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def partition_similarity(
    part_a: Mapping[Hashable, Hashable], part_b: Mapping[Hashable, Hashable]
) -> dict[str, float | int]:
    r"""Compare two community partitions over their shared nodes.

    Parameters
    ----------
    part_a, part_b : mapping
        ``node -> community id``.

    Returns
    -------
    dict
        ``n_common``, ``ari`` (Adjusted Rand Index) and ``nmi`` (normalized
        mutual information); both NaN with fewer than 2 shared nodes.

    Notes
    -----
    From the contingency table :math:`n_{ij}` with marginals :math:`a_i, b_j`,

    .. math::

        \mathrm{ARI} = \frac{S - E}{(S_a + S_b)/2 - E}, \quad
        S = \sum_{ij}\binom{n_{ij}}{2}, \;
        S_a = \sum_i\binom{a_i}{2}, \;
        S_b = \sum_j\binom{b_j}{2}, \;
        E = \frac{S_a S_b}{\binom{n}{2}}

    (Hubert and Arabie, 1985), and
    :math:`\mathrm{NMI} = 2 I(A;B) / (H(A) + H(B))` with natural logarithms.
    ARI is 1 when its denominator vanishes; NMI is 1 when both partitions
    are trivial.
    """
    nodes = sorted(set(part_a) & set(part_b), key=str)
    n = len(nodes)
    if n < MIN_COMMON_NODES:
        return {"n_common": n, "ari": float("nan"), "nmi": float("nan")}

    table = _contingency(part_a, part_b, nodes)
    rows, cols = table.sum(axis=1), table.sum(axis=0)

    def pairs(values: np.ndarray) -> float:
        return float((values * (values - 1) / 2).sum())

    sum_cells, sum_rows, sum_cols = pairs(table), pairs(rows), pairs(cols)
    expected = sum_rows * sum_cols / (n * (n - 1) / 2)
    max_index = (sum_rows + sum_cols) / 2
    ari = (sum_cells - expected) / (max_index - expected) if max_index != expected else 1.0

    p_ab = table / n
    p_a, p_b = rows / n, cols / n
    nonzero = p_ab > 0
    mutual = float((p_ab[nonzero] * np.log(p_ab[nonzero] / np.outer(p_a, p_b)[nonzero])).sum())
    h_a, h_b = _entropy(p_a), _entropy(p_b)
    nmi = 1.0 if h_a == 0.0 and h_b == 0.0 else 2 * mutual / (h_a + h_b)
    return {"n_common": n, "ari": float(ari), "nmi": float(nmi)}


def community_crosstab(
    part_a: Mapping[Hashable, Hashable], part_b: Mapping[Hashable, Hashable]
) -> pd.DataFrame:
    """Cross-tabulate two partitions over their shared nodes.

    Parameters
    ----------
    part_a, part_b : mapping
        ``node -> community id``.

    Returns
    -------
    pandas.DataFrame
        Communities of A as rows, of B as columns, shared-node counts as cells.
    """
    nodes = sorted(set(part_a) & set(part_b), key=str)
    if not nodes:
        return pd.DataFrame()
    return pd.crosstab(
        pd.Series([part_a[n] for n in nodes], name="A"),
        pd.Series([part_b[n] for n in nodes], name="B"),
    )


def community_shared_nodes(
    part_a: Mapping[Hashable, Hashable], part_b: Mapping[Hashable, Hashable]
) -> dict[tuple[Hashable, Hashable], list[Hashable]]:
    """List the shared nodes behind each cell of :func:`community_crosstab`.

    Parameters
    ----------
    part_a, part_b : mapping
        ``node -> community id``.

    Returns
    -------
    dict
        ``(community_a, community_b) -> nodes``, nodes sorted as strings.
    """
    out: dict[tuple[Hashable, Hashable], list[Hashable]] = {}
    for node in sorted(set(part_a) & set(part_b), key=str):
        out.setdefault((part_a[node], part_b[node]), []).append(node)
    return out


def _attr_keys(labels: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i, label in enumerate(labels):
        key = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") or "e"
        out[label] = f"{key}_{i}" if key in out.values() else key
    return out


def union_graph(graphs: Mapping[str, nx.Graph]) -> nx.Graph:
    """Merge several graphs into one, annotating where each node and edge occurs.

    Parameters
    ----------
    graphs : mapping of str to networkx.Graph
        At least one graph.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        Directed only when every input is directed. Node attributes:
        ``n_graphs``, ``membership`` and ``in_<label>`` (0/1). Edge
        attributes: ``weight_<label>`` (normalized weight, 0 when absent),
        ``weight`` (their mean), ``n_graphs`` and ``membership``. Membership
        is ``ONLY_A``/``ONLY_B``/``BOTH`` for two graphs and ``"k/N"``
        otherwise. ``<label>`` is reduced to ``[A-Za-z0-9_]`` so the graph can
        be written as GML; colliding labels get a positional suffix.

    Raises
    ------
    ValueError
        If ``graphs`` is empty.
    """
    if not graphs:
        msg = "union_graph requires at least one graph"
        raise ValueError(msg)

    labels = list(graphs)
    directed = all(g.is_directed() for g in graphs.values())
    union: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    keys = _attr_keys(labels)
    node_sets = {label: set(g.nodes()) for label, g in graphs.items()}
    weights = {label: _normalized(canonical_edges(g)) for label, g in graphs.items()}

    def membership(present: list[str]) -> str:
        if len(labels) == TWO:
            if len(present) == TWO:
                return BOTH
            return ONLY_A if present[0] == labels[0] else ONLY_B
        return f"{len(present)}/{len(labels)}"

    for node in sorted(set().union(*node_sets.values()), key=str):
        present = [label for label in labels if node in node_sets[label]]
        flags = {f"in_{keys[label]}": int(label in present) for label in labels}
        union.add_node(node, n_graphs=len(present), membership=membership(present), **flags)

    for edge in sorted(set().union(*weights.values()), key=str):
        present = [label for label in labels if edge in weights[label]]
        edge_weights = {f"weight_{keys[label]}": weights[label].get(edge, 0.0) for label in labels}
        union.add_edge(
            *edge,
            weight=float(np.mean(list(edge_weights.values()))),
            n_graphs=len(present),
            membership=membership(present),
            **edge_weights,
        )
    return union


def rarefied_profiles(
    build_graph: BuildGraph,
    informant_ids_by_label: Mapping[str, Sequence[Hashable]],
    *,
    k: int = 10,
    seed: int = 42,
    progress_cb: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Compare structural profiles with group sizes equalized by subsampling.

    Parameters
    ----------
    build_graph : callable
        Builds a graph from a list of informant ids.
    informant_ids_by_label : mapping of str to sequence
        Informants behind each graph.
    k : int, default 10
        Subsamples per label.
    seed : int, default 42
        Seed for :func:`numpy.random.default_rng`.
    progress_cb : callable, optional
        Called with ``(done, len(labels) * k)`` after each subsample.

    Returns
    -------
    pandas.DataFrame
        A ``metric`` column with ``RAREFACTION_METRICS`` and, per label,
        ``<label> (mean)`` and ``<label> (sd)`` (ddof 1) over the ``k``
        subsamples. ``attrs`` holds ``n_min`` and ``k``.

    Raises
    ------
    ValueError
        If ``k < 1`` or no labels are given.

    Notes
    -----
    Every label is subsampled without replacement to the size of the
    smallest group, :math:`n_{min}`, so differences reflect structure rather
    than the number of informants. Costs ``len(labels) * k`` graph builds.
    """
    if k < 1:
        msg = "k must be at least 1"
        raise ValueError(msg)
    labels = list(informant_ids_by_label)
    if not labels:
        msg = "rarefied_profiles requires at least one label"
        raise ValueError(msg)

    n_min = min(len(ids) for ids in informant_ids_by_label.values())
    rng = np.random.default_rng(seed)
    total = len(labels) * k
    out: dict[str, Any] = {"metric": list(RAREFACTION_METRICS)}
    for li, label in enumerate(labels):
        ids = np.asarray(list(informant_ids_by_label[label]), dtype=object)
        samples = np.empty((k, len(RAREFACTION_METRICS)))
        for i in range(k):
            chosen = rng.choice(ids, size=n_min, replace=False) if len(ids) > n_min else ids
            profile = _base_profile(build_graph(list(chosen)))
            samples[i] = [
                np.nan if profile[m] is None else float(profile[m]) for m in RAREFACTION_METRICS
            ]
            if progress_cb:
                progress_cb(li * k + i + 1, total)
        defined = ~np.isnan(samples).all(axis=0)
        means = np.full(len(RAREFACTION_METRICS), np.nan)
        sds = np.zeros(len(RAREFACTION_METRICS))
        means[defined] = np.nanmean(samples[:, defined], axis=0)
        if k > 1:
            sds[defined] = np.nanstd(samples[:, defined], axis=0, ddof=1)
        out[f"{label} (mean)"] = means
        out[f"{label} (sd)"] = sds

    result = pd.DataFrame(out)
    result.attrs["n_min"] = n_min
    result.attrs["k"] = k
    return result


def permutation_overlap_test(
    build_graph: BuildGraph,
    ids_a: Sequence[Hashable],
    ids_b: Sequence[Hashable],
    *,
    k: int = 100,
    seed: int = 42,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    r"""Test whether two groups' graphs overlap less than expected by chance.

    Parameters
    ----------
    build_graph : callable
        Builds a graph from a list of informant ids.
    ids_a, ids_b : sequence
        Informants of each group.
    k : int, default 100
        Permutations of the group label.
    seed : int, default 42
        Seed for :func:`numpy.random.default_rng`.
    progress_cb : callable, optional
        Called with ``(i, k)`` before each permutation and ``(k, k)`` at the end.

    Returns
    -------
    dict
        ``jaccard_nodes`` and ``jaccard_edges``, each with ``observed``,
        ``null`` (array of ``k`` permuted values), ``null_mean``, ``null_sd``
        (ddof 1) and ``p_value``; plus ``k``, ``n_a`` and ``n_b``.

    Raises
    ------
    ValueError
        If ``k < 1`` or either group is empty.

    Notes
    -----
    The pooled informants are shuffled and split back into groups of the
    original sizes. The one-sided p-value is
    :math:`(\#\{J_{null} \le J_{obs}\} + 1) / (k + 1)`. Costs ``2k + 2``
    graph builds.
    """
    if k < 1:
        msg = "k must be at least 1"
        raise ValueError(msg)
    ids_a, ids_b = list(ids_a), list(ids_b)
    if not ids_a or not ids_b:
        msg = "both groups need at least one informant"
        raise ValueError(msg)

    def jaccards(graph_a: nx.Graph, graph_b: nx.Graph) -> tuple[float, float]:
        nodes = _set_metrics(set(graph_a.nodes()), set(graph_b.nodes()))["jaccard"]
        edges = _set_metrics(set(canonical_edges(graph_a)), set(canonical_edges(graph_b)))[
            "jaccard"
        ]
        return float(nodes), float(edges)

    observed = jaccards(build_graph(ids_a), build_graph(ids_b))
    pool = np.asarray(ids_a + ids_b, dtype=object)
    n_a = len(ids_a)
    rng = np.random.default_rng(seed)
    null = np.empty((k, 2))
    for i in range(k):
        if progress_cb:
            progress_cb(i, k)
        shuffled = rng.permutation(pool)
        null[i] = jaccards(build_graph(list(shuffled[:n_a])), build_graph(list(shuffled[n_a:])))
    if progress_cb:
        progress_cb(k, k)

    def summary(obs: float, values: np.ndarray) -> dict[str, Any]:
        return {
            "observed": obs,
            "null": values,
            "null_mean": float(values.mean()),
            "null_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "p_value": float((np.sum(values <= obs) + 1) / (values.size + 1)),
        }

    return {
        "jaccard_nodes": summary(observed[0], null[:, 0].copy()),
        "jaccard_edges": summary(observed[1], null[:, 1].copy()),
        "k": k,
        "n_a": n_a,
        "n_b": len(ids_b),
    }
