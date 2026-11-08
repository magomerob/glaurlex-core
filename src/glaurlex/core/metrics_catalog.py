"""Metadata for every metric computed by the package.

``METRIC_CATALOG`` is the single source of truth for metric keys, descriptions
and the function that computes each one. Display labels in other languages are
expected to be keyed by ``MetricSpec.key``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MetricScope = Literal["informant", "type", "node", "comparison"]


@dataclass(frozen=True)
class MetricSpec:
    """Description of one metric.

    Attributes
    ----------
    key : str
        Stable identifier; also the column or dictionary key in the output.
    name : str
        Short English name.
    description : str
        One-sentence definition.
    scope : {"informant", "type", "node", "comparison"}
        Unit the metric is computed for.
    algorithm : str
        Formula or library call.
    parameters : str
        Fixed parameters of the computation.
    implementation : str
        Dotted path of the function that computes it.
    """

    key: str
    name: str
    description: str
    scope: MetricScope
    algorithm: str
    parameters: str = ""
    implementation: str = ""


_INFORMANT = "glaurlex.core.inference.informant_metrics"
_TYPE = "glaurlex.core.stats.type_statistics"
_NODE = "glaurlex.core.graph.node_stats"

_SPECS = (
    MetricSpec(
        "n_tokens",
        "Tokens produced",
        "Number of responses produced by the informant.",
        "informant",
        "Row count per informant in the topic frame.",
        implementation=_INFORMANT,
    ),
    MetricSpec(
        "n_types",
        "Distinct types",
        "Number of distinct types produced by the informant.",
        "informant",
        "Distinct values of `type` per informant.",
        implementation=_INFORMANT,
    ),
    MetricSpec(
        "type_coverage",
        "Type coverage",
        "Share of the topic's distinct types produced by the informant.",
        "informant",
        "n_types / distinct types in the topic.",
        implementation=_INFORMANT,
    ),
    MetricSpec(
        "ttr",
        "Type-token ratio",
        "Lexical diversity of the informant's list.",
        "informant",
        "n_types / n_tokens.",
        implementation=_INFORMANT,
    ),
    MetricSpec(
        "mean_pos",
        "Mean position",
        "Mean 1-indexed position of the informant's responses.",
        "informant",
        "mean(pos) + 1.",
        implementation=_INFORMANT,
    ),
    MetricSpec(
        "max_pos",
        "List length",
        "Last position reached by the informant.",
        "informant",
        "max(pos) + 1.",
        implementation=_INFORMANT,
    ),
    MetricSpec(
        "entropy",
        "Shannon entropy",
        "Entropy of the informant's type distribution.",
        "informant",
        "H = -sum(p_t * log2(p_t)) over type proportions.",
        parameters="Base 2 (bits).",
        implementation=_INFORMANT,
    ),
    MetricSpec(
        "total_availability",
        "Total availability",
        "Sum of the availability of the tokens produced by the informant.",
        "informant",
        "Sum over the informant's tokens of the type's `availability`.",
        parameters="Availability is computed over the analysed group, not globally.",
        implementation=_INFORMANT,
    ),
    MetricSpec(
        "mean_availability",
        "Mean availability",
        "Mean availability of the tokens produced by the informant.",
        "informant",
        "Mean over the informant's tokens of the type's `availability`.",
        parameters="Availability is computed over the analysed group, not globally.",
        implementation=_INFORMANT,
    ),
    MetricSpec(
        "availability",
        "Lexical availability",
        "Position-weighted share of informants who produced the type.",
        "type",
        "sum_i exp(-2.3 * i / max_pos) * informants_at_position_i / informants.",
        parameters="Exponential decay factor 2.3, normalized by the topic's highest position.",
        implementation=_TYPE,
    ),
    MetricSpec(
        "appearance",
        "Appearance",
        "Share of informants who produced the type at least once.",
        "type",
        "Distinct informants producing the type / informants.",
        implementation=_TYPE,
    ),
    MetricSpec(
        "rel_freq",
        "Relative frequency",
        "Share of the topic's tokens that are this type.",
        "type",
        "value_counts(normalize=True) over `type`.",
        implementation=_TYPE,
    ),
    MetricSpec(
        "avg_pos",
        "Average position",
        "Mean 0-indexed position at which the type is produced.",
        "type",
        "mean(pos) per type.",
        implementation=_TYPE,
    ),
    MetricSpec(
        "cum_freq",
        "Cumulative frequency",
        "Cumulative relative frequency with types sorted by availability.",
        "type",
        "Running sum of `rel_freq` in descending `availability` order.",
        implementation=_TYPE,
    ),
    MetricSpec(
        "tokens",
        "Tokens",
        "Occurrences of the type in the topic.",
        "type",
        "value_counts() over `type`.",
        implementation=_TYPE,
    ),
    MetricSpec(
        "degree",
        "Degree",
        "Number of neighbours of the node.",
        "node",
        "Unweighted degree.",
        implementation=_NODE,
    ),
    MetricSpec(
        "degree_centrality",
        "Degree centrality",
        "Degree normalized by V - 1.",
        "node",
        "nx.degree_centrality.",
        implementation=_NODE,
    ),
    MetricSpec(
        "strength",
        "Strength",
        "Sum of the weights of the node's edges.",
        "node",
        'Degree with weight="weight".',
        implementation=_NODE,
    ),
    MetricSpec(
        "betweenness",
        "Betweenness",
        "Number of shortest paths through the node.",
        "node",
        "nx.betweenness_centrality, unnormalized.",
        parameters="normalized=False",
        implementation=_NODE,
    ),
    MetricSpec(
        "closeness",
        "Closeness",
        "Inverse mean distance to the reachable nodes.",
        "node",
        "nx.closeness_centrality with the Wasserman-Faust correction.",
        parameters="wf_improved=True",
        implementation=_NODE,
    ),
    MetricSpec(
        "pagerank",
        "PageRank",
        "Stationary probability of a weighted random walk with restarts.",
        "node",
        "nx.pagerank.",
        parameters='alpha=0.85, tol=1e-6, weight="weight"',
        implementation=_NODE,
    ),
    MetricSpec(
        "eigenvector",
        "Eigenvector centrality",
        "Node entry of the leading eigenvector of the weighted adjacency matrix.",
        "node",
        "nx.eigenvector_centrality.",
        parameters='max_iter=1000, weight="weight"',
        implementation=_NODE,
    ),
    MetricSpec(
        "clustering",
        "Clustering",
        "Weighted local clustering coefficient.",
        "node",
        "nx.clustering.",
        parameters='weight="weight"',
        implementation=_NODE,
    ),
    MetricSpec(
        "jaccard_nodes",
        "Jaccard (nodes)",
        "Share of the types in either graph that occur in both.",
        "comparison",
        "|A ∩ B| / |A ∪ B| over node sets.",
        parameters="0: no shared type. 1: identical vocabularies.",
        implementation="glaurlex.core.graph_compare.compare_pair_nodes",
    ),
    MetricSpec(
        "jaccard_edges",
        "Jaccard (edges)",
        "Share of the type pairs in either graph that occur in both.",
        "comparison",
        "|A ∩ B| / |A ∪ B| over canonical edge sets.",
        parameters="Usually lower than the node index: shared words can be chained differently.",
        implementation="glaurlex.core.graph_compare.compare_pair_edges",
    ),
    MetricSpec(
        "overlap_coef",
        "Overlap coefficient",
        "Shared types relative to the smaller vocabulary.",
        "comparison",
        "|A ∩ B| / min(|A|, |B|).",
        parameters="Unlike Jaccard, not penalized by a size difference.",
        implementation="glaurlex.core.graph_compare.compare_pair_nodes",
    ),
    MetricSpec(
        "dice",
        "Dice coefficient",
        "Overlap of the node sets.",
        "comparison",
        "2 |A ∩ B| / (|A| + |B|).",
        implementation="glaurlex.core.graph_compare.compare_pair_nodes",
    ),
    MetricSpec(
        "modularity",
        "Modularity",
        "Concentration of edge weight inside communities.",
        "comparison",
        'nx.community.modularity(undirected graph, communities, weight="weight").',
        implementation="glaurlex.core.graph_compare.graph_profile",
    ),
    MetricSpec(
        "largest_component_fraction",
        "Largest component fraction",
        "Share of nodes in the largest connected component.",
        "comparison",
        "Largest (weakly) connected component size / V.",
        implementation="glaurlex.core.graph_compare.graph_profile",
    ),
    MetricSpec(
        "ari",
        "Adjusted Rand index",
        "Agreement between two community partitions, corrected for chance.",
        "comparison",
        "ARI from the contingency table over shared nodes.",
        parameters="1: identical partitions. Near 0: chance agreement.",
        implementation="glaurlex.core.graph_compare.partition_similarity",
    ),
    MetricSpec(
        "nmi",
        "Normalized mutual information",
        "Shared information between two community partitions.",
        "comparison",
        "2 I(A;B) / (H(A) + H(B)) over shared nodes.",
        parameters="0: independent partitions. 1: one determines the other.",
        implementation="glaurlex.core.graph_compare.partition_similarity",
    ),
    MetricSpec(
        "spearman_rho",
        "Spearman's rho",
        "Rank agreement of a node metric over shared nodes.",
        "comparison",
        "scipy.stats.spearmanr.",
        implementation="glaurlex.core.graph_compare.rank_correlation",
    ),
    MetricSpec(
        "kendall_tau_b",
        "Kendall's tau-b",
        "Rank agreement of a node metric over shared nodes, adjusted for ties.",
        "comparison",
        'scipy.stats.kendalltau(variant="b").',
        implementation="glaurlex.core.graph_compare.rank_correlation",
    ),
)

METRIC_CATALOG: dict[str, MetricSpec] = {spec.key: spec for spec in _SPECS}


def metrics_by_scope(scope: MetricScope) -> dict[str, MetricSpec]:
    """Return the metrics of one scope in catalog order.

    Parameters
    ----------
    scope : {"informant", "type", "node", "comparison"}

    Returns
    -------
    dict
        ``key -> MetricSpec``.
    """
    return {k: m for k, m in METRIC_CATALOG.items() if m.scope == scope}


def labels_by_scope(scope: MetricScope) -> dict[str, str]:
    """Return ``key -> name`` for the metrics of one scope, in catalog order.

    Parameters
    ----------
    scope : {"informant", "type", "node", "comparison"}

    Returns
    -------
    dict
    """
    return {k: m.name for k, m in metrics_by_scope(scope).items()}
