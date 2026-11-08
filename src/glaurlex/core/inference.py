"""Per-informant lexical metrics and inference against sociolinguistic variables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from glaurlex.core.stats import type_statistics

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

ALPHA = 0.05
SKEW_MIN_N = 3
KURTOSIS_MIN_N = 4
SHAPIRO_MIN_N = 3
SHAPIRO_MAX_N = 5000
MIN_N_PER_GROUP = 2
MIN_GROUPS = 2
MIN_GROUPS_FOR_POSTHOC = 3
MIN_N_FOR_TREND = 3
MIN_N_FOR_CORRELATION = 3

INSUFFICIENT_DATA = "insufficient_data"


def _sorted_sum(values: pd.Series, informant_ids: pd.Index | pd.Series) -> pd.Series:
    frame = pd.DataFrame({"informant_id": np.asarray(informant_ids), "value": values.to_numpy()})
    return frame.sort_values(["informant_id", "value"]).groupby("informant_id")["value"].sum()


def informant_metrics(
    df: pd.DataFrame,
    informants: pd.DataFrame,
    type_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    r"""Compute lexical metrics for every informant present in a topic frame.

    Parameters
    ----------
    df : pandas.DataFrame
        Topic frame (``informant_id``, ``pos``, ``type``), already restricted
        to the informants of interest.
    informants : pandas.DataFrame
        Informants frame keyed by ``informant_id``; its other columns are
        appended to the result.
    type_stats : pandas.DataFrame, optional
        Output of :func:`~glaurlex.core.stats.type_statistics` for ``df``.
        Computed when omitted.

    Returns
    -------
    pandas.DataFrame
        One row per informant, sorted by ``informant_id``, with:

        * ``n_tokens``: responses produced; ``n_types``: distinct types;
        * ``type_coverage``: ``n_types`` / distinct types in the topic;
        * ``ttr``: ``n_types / n_tokens``;
        * ``mean_pos``, ``max_pos``: mean and last position, **1-indexed**;
        * ``entropy``: Shannon entropy of the type distribution, in bits;
        * ``total_availability``, ``mean_availability``: sum and mean of the
          availability of each produced token.

        Availability is relative to ``df``: restricting ``df`` to a group
        changes these two columns.

    Raises
    ------
    ValueError
        If ``informants`` has no ``informant_id`` column.

    Notes
    -----
    Entropy is :math:`H = -\sum_t p_t \log_2 p_t` over the informant's type
    proportions. Entropy and availability terms are summed in ascending
    order, so informants with the same responses in any order tie exactly,
    which rank-based tests depend on. Runs in :math:`O(R \log R)` for
    :math:`R` responses.
    """
    if "informant_id" not in informants.columns:
        msg = "informants must have an 'informant_id' column"
        raise ValueError(msg)
    if df.empty:
        return pd.DataFrame()

    if type_stats is None:
        type_stats = type_statistics(df)
    availability = (
        type_stats.set_index("type")["availability"]
        if "availability" in type_stats.columns
        else pd.Series(dtype=float)
    )

    work = df[["informant_id", "pos", "type"]].assign(
        availability=df["type"].map(availability).astype(float).fillna(0.0)
    )
    by_informant = work.groupby("informant_id")
    type_counts = work.groupby(["informant_id", "type"]).size()
    shares = type_counts / type_counts.groupby(level="informant_id").transform("sum")
    entropy = _sorted_sum(
        -(shares * np.log2(shares)), shares.index.get_level_values("informant_id")
    )
    total_availability = _sorted_sum(work["availability"], work["informant_id"])

    metrics = pd.DataFrame(
        {
            "n_tokens": by_informant.size(),
            "n_types": by_informant["type"].nunique(),
            "mean_pos": by_informant["pos"].mean() + 1,
            "max_pos": by_informant["pos"].max() + 1,
            "entropy": entropy,
            "total_availability": total_availability,
            "mean_availability": total_availability / by_informant.size(),
        }
    )
    metrics.insert(2, "type_coverage", metrics["n_types"] / df["type"].nunique())
    metrics.insert(3, "ttr", metrics["n_types"] / metrics["n_tokens"])
    metrics = metrics.reset_index()

    return metrics.merge(informants, on="informant_id", how="left")


def describe_series(s: pd.Series) -> dict[str, float | int]:
    """Summarize a numeric series.

    Parameters
    ----------
    s : pandas.Series
        Values are coerced to numbers; non-numeric entries are dropped.

    Returns
    -------
    dict
        ``n``, ``mean``, ``std`` (ddof 1), ``se``, ``min``, ``p5``, ``q1``,
        ``median``, ``q3``, ``p95``, ``max``, ``iqr``, ``range``,
        ``variance`` (ddof 1), ``cv`` (``std / mean``), ``skewness`` and
        ``kurtosis`` (bias-corrected; excess kurtosis). Only ``n`` is returned
        for an empty series. ``skewness`` needs at least 3 values and
        ``kurtosis`` at least 4; otherwise they are NaN.
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    n = int(s.shape[0])
    if n == 0:
        return {"n": 0}

    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else 0.0
    p5, q1, median, q3, p95 = np.percentile(s, [5, 25, 50, 75, 95])
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": std / np.sqrt(n),
        "min": float(s.min()),
        "p5": float(p5),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "p95": float(p95),
        "max": float(s.max()),
        "iqr": float(q3 - q1),
        "range": float(s.max() - s.min()),
        "variance": float(s.var(ddof=1)) if n > 1 else 0.0,
        "cv": (std / mean) if mean != 0 else np.nan,
        "skewness": float(stats.skew(s, bias=False)) if n >= SKEW_MIN_N else np.nan,
        "kurtosis": float(stats.kurtosis(s, bias=False)) if n >= KURTOSIS_MIN_N else np.nan,
    }


def normality_test(s: pd.Series, alpha: float = ALPHA) -> dict[str, Any]:
    r"""Test a numeric series for normality.

    Parameters
    ----------
    s : pandas.Series
        Values are coerced to numbers; non-numeric entries are dropped.
    alpha : float, default 0.05
        Significance level for ``is_normal``.

    Returns
    -------
    dict
        ``test``, ``n``, ``statistic``, ``p_value`` and ``is_normal``
        (``p_value > alpha``); or ``test``, ``n`` and ``error`` when fewer than
        3 values remain.

    Notes
    -----
    Shapiro-Wilk for :math:`n \le 5000`, D'Agostino-Pearson :math:`K^2` above.
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    n = int(s.shape[0])
    if n < SHAPIRO_MIN_N:
        return {"test": "Shapiro-Wilk", "n": n, "error": INSUFFICIENT_DATA}
    if n > SHAPIRO_MAX_N:
        name = "D'Agostino-Pearson"
        statistic, p = stats.normaltest(s)
    else:
        name = "Shapiro-Wilk"
        statistic, p = stats.shapiro(s)
    return {
        "test": name,
        "n": n,
        "statistic": float(statistic),
        "p_value": float(p),
        "is_normal": bool(p > alpha),
    }


@dataclass(frozen=True)
class QQPlot:
    """Normal Q-Q plot coordinates and least-squares fit.

    Attributes
    ----------
    theoretical, observed : numpy.ndarray
        Normal quantiles and the sorted sample.
    slope, intercept, r : float
        Fit of ``observed`` on ``theoretical`` and its correlation coefficient.
    """

    theoretical: np.ndarray
    observed: np.ndarray
    slope: float
    intercept: float
    r: float


def qq_points(s: pd.Series) -> QQPlot:
    """Compute normal Q-Q plot coordinates for a numeric series.

    Parameters
    ----------
    s : pandas.Series
        Values are coerced to numbers; non-numeric entries are dropped.

    Returns
    -------
    QQPlot
        Filliben plotting positions, as :func:`scipy.stats.probplot`.
    """
    values = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    (theoretical, observed), (slope, intercept, r) = stats.probplot(values, dist="norm")
    return QQPlot(
        theoretical=np.asarray(theoretical),
        observed=np.asarray(observed),
        slope=float(slope),
        intercept=float(intercept),
        r=float(r),
    )


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < MIN_N_PER_GROUP or nb < MIN_N_PER_GROUP:
        return float("nan")
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def _rank_biserial(u: float, n1: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return float("nan")
    return float(1.0 - (2.0 * u) / (n1 * n2))


def _eta_squared(groups: Sequence[np.ndarray]) -> float:
    values = np.concatenate(groups)
    grand_mean = values.mean()
    ss_total = float(((values - grand_mean) ** 2).sum())
    if ss_total == 0:
        return float("nan")
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    return float(ss_between / ss_total)


def _kruskal(*groups: np.ndarray) -> tuple[float, float]:
    try:
        h, p = stats.kruskal(*groups)
    except ValueError:
        return float("nan"), float("nan")
    return float(h), float(p)


def _epsilon_squared(h: float, n: int, k: int) -> float:
    if n <= k:
        return float("nan")
    return float((h - k + 1) / (n - k))


@dataclass
class InferenceResult:
    """Outcome of :func:`compare_groups`.

    Attributes
    ----------
    metric, by : str
        Compared column and grouping column.
    kind : str
        ``"two-group"``, ``"k-group"`` or ``"insufficient"``.
    n_total : int
        Observations in the groups that entered the tests.
    descriptives : pandas.DataFrame
        Per-level summary from which the tests were run.
    parametric, non_parametric, effect_size : dict
        Test statistics and effect sizes; empty when ``kind`` is ``"insufficient"``.
    posthoc : pandas.DataFrame or None
        Pairwise Mann-Whitney comparisons (3+ groups only).
    trend : dict or None
        Rank correlation between level order and metric (ordinal variables only).
    notes : list of str
        Codes for undefined statistics: ``"insufficient_levels"``,
        ``"welch_t_undefined"``, ``"anova_undefined"``, ``"kruskal_undefined"``.
    """

    metric: str
    by: str
    kind: str
    n_total: int
    descriptives: pd.DataFrame
    parametric: dict[str, Any] = field(default_factory=dict)
    non_parametric: dict[str, Any] = field(default_factory=dict)
    effect_size: dict[str, float] = field(default_factory=dict)
    posthoc: pd.DataFrame | None = None
    trend: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


def _ordered_levels(levels: Sequence[Hashable], order: Sequence[object] | None) -> list[Hashable]:
    if not order:
        return sorted(levels, key=str)
    by_str = {str(level): level for level in levels}
    declared = [by_str[str(o)] for o in order if str(o) in by_str]
    declared_str = {str(level) for level in declared}
    extras = sorted((lv for k, lv in by_str.items() if k not in declared_str), key=str)
    return declared + extras


def _trend_tests(
    work: pd.DataFrame, metric: str, by: str, ordered_levels: Sequence[Hashable]
) -> dict[str, Any]:
    rank = work[by].map({level: i for i, level in enumerate(ordered_levels)})
    sub = pd.DataFrame({"rank": rank, "value": work[metric]}).dropna()
    n = int(sub.shape[0])
    if n < MIN_N_FOR_TREND or sub["rank"].nunique() < MIN_GROUPS:
        return {"n": n, "error": INSUFFICIENT_DATA}

    rho, p_rho = stats.spearmanr(sub["rank"], sub["value"])
    tau, p_tau = stats.kendalltau(sub["rank"], sub["value"], variant="b")
    return {
        "n": n,
        "levels": [str(level) for level in ordered_levels],
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "kendall_tau_b": float(tau),
        "kendall_p": float(p_tau),
    }


def nonparametric_p_value(groups: Sequence[np.ndarray]) -> float:
    """Return the p-value of the non-parametric omnibus test across groups.

    Parameters
    ----------
    groups : sequence of numpy.ndarray
        Metric values per group. Groups with fewer than two values are ignored.

    Returns
    -------
    float
        Two-sided Mann-Whitney U for two groups, Kruskal-Wallis H for three
        or more; NaN with fewer than two usable groups.
    """
    usable = [g for g in groups if len(g) >= MIN_N_PER_GROUP]
    if len(usable) < MIN_GROUPS:
        return float("nan")
    if len(usable) == MIN_GROUPS:
        return float(stats.mannwhitneyu(usable[0], usable[1], alternative="two-sided").pvalue)
    return _kruskal(*usable)[1]


def _group_descriptives(
    work: pd.DataFrame, metric: str, by: str, order: Sequence[Hashable] | None
) -> pd.DataFrame:
    grouped = work.groupby(by)[metric]
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "mean": grouped.mean(),
            "std": grouped.std(ddof=1).fillna(0.0),
            "median": grouped.median(),
            "q1": grouped.quantile(0.25),
            "q3": grouped.quantile(0.75),
            "min": grouped.min().astype(float),
            "max": grouped.max().astype(float),
        }
    ).reset_index()
    if out.empty or not order:
        return out.sort_values(by).reset_index(drop=True)
    rank = {str(level): i for i, level in enumerate(order)}
    position = out[by].map(lambda v: rank.get(str(v), len(rank)))
    return (
        out.assign(_rank=position)
        .sort_values(["_rank", by])
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def compare_groups(
    df: pd.DataFrame,
    metric: str,
    by: str,
    *,
    posthoc: bool = True,
    posthoc_alpha: float = ALPHA,
    order: Sequence[object] | None = None,
) -> InferenceResult:
    r"""Compare a numeric metric across the levels of a categorical variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Contains ``metric`` and ``by``.
    metric : str
        Numeric column; values are coerced to numbers.
    by : str
        Grouping column. Levels with fewer than two observations are excluded
        from the tests.
    posthoc : bool, default True
        Run pairwise comparisons when there are three or more groups.
    posthoc_alpha : float, default 0.05
        Significance level applied to the Bonferroni-adjusted p-values.
    order : sequence, optional
        Declared level order of an ordinal variable. Orders the descriptives
        and post-hoc table, and adds a trend test. Levels missing from
        ``order`` follow in alphabetical order.

    Returns
    -------
    InferenceResult

    Notes
    -----
    * Two groups: Welch's t-test, two-sided Mann-Whitney U, Cohen's d with
      pooled standard deviation and rank-biserial
      :math:`r = 1 - 2U / (n_1 n_2)`. ``parametric["df_pooled"]`` reports
      :math:`n_1 + n_2 - 2`, not the Welch-Satterthwaite degrees of freedom.
    * Three or more groups: one-way ANOVA, Kruskal-Wallis H,
      :math:`\eta^2 = SS_{between} / SS_{total}` and
      :math:`\varepsilon^2 = (H - k + 1) / (n - k)`; post-hoc Mann-Whitney U
      with Bonferroni correction over the :math:`k(k-1)/2` pairs.
    * Trend: Spearman :math:`\rho` and Kendall :math:`\tau_b` between the
      level rank and the metric, over levels with at least one observation.
    """
    work = df[[metric, by]].assign(**{metric: pd.to_numeric(df[metric], errors="coerce")})
    work = work.dropna(subset=[metric, by])

    levels = _ordered_levels(work[by].unique().tolist(), order)
    values = {level: work.loc[work[by] == level, metric].to_numpy() for level in levels}
    tested = [level for level in levels if len(values[level]) >= MIN_N_PER_GROUP]
    groups = [values[level] for level in tested]

    desc = _group_descriptives(work, metric, by, levels if order else None)
    trend = None
    if order:
        observed = [level for level in levels if len(values[level]) >= 1]
        if len(observed) >= MIN_GROUPS:
            trend = _trend_tests(work, metric, by, observed)

    if len(groups) < MIN_GROUPS:
        return InferenceResult(
            metric=metric,
            by=by,
            kind="insufficient",
            n_total=len(work),
            descriptives=desc,
            notes=["insufficient_levels"],
        )

    n_total = int(sum(len(g) for g in groups))
    notes: list[str] = []

    if len(groups) == MIN_GROUPS:
        a, b = groups
        t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
        if np.isnan(t_p):
            notes.append("welch_t_undefined")
        return InferenceResult(
            metric=metric,
            by=by,
            kind="two-group",
            n_total=n_total,
            descriptives=desc,
            parametric={
                "test": "Welch's t-test",
                "statistic": float(t_stat),
                "p_value": float(t_p),
                "df_pooled": float(len(a) + len(b) - 2),
            },
            non_parametric={
                "test": "Mann-Whitney U (two-sided)",
                "U": float(u_stat),
                "p_value": float(u_p),
            },
            effect_size={
                "cohens_d": _cohens_d(a, b),
                "rank_biserial_r": _rank_biserial(float(u_stat), len(a), len(b)),
            },
            trend=trend,
            notes=notes,
        )

    f_stat, f_p = stats.f_oneway(*groups)
    h_stat, h_p = _kruskal(*groups)
    if np.isnan(f_p):
        notes.append("anova_undefined")
    if np.isnan(h_p):
        notes.append("kruskal_undefined")

    posthoc_df = None
    if posthoc and len(groups) >= MIN_GROUPS_FOR_POSTHOC:
        n_pairs = len(tested) * (len(tested) - 1) // 2
        rows = []
        for i, level_1 in enumerate(tested):
            for level_2 in tested[i + 1 :]:
                a, b = values[level_1], values[level_2]
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                p_adj = min(p * n_pairs, 1.0) if not np.isnan(p) else float("nan")
                rows.append(
                    {
                        "group_1": level_1,
                        "group_2": level_2,
                        "n_1": len(a),
                        "n_2": len(b),
                        "U": float(u),
                        "p_value": float(p),
                        "p_bonferroni": float(p_adj),
                        "significant": bool(p_adj < posthoc_alpha),
                        "rank_biserial_r": _rank_biserial(float(u), len(a), len(b)),
                    }
                )
        posthoc_df = pd.DataFrame(rows)

    return InferenceResult(
        metric=metric,
        by=by,
        kind="k-group",
        n_total=n_total,
        descriptives=desc,
        parametric={
            "test": "One-way ANOVA",
            "F": float(f_stat),
            "p_value": float(f_p),
            "df_between": float(len(groups) - 1),
            "df_within": float(n_total - len(groups)),
        },
        non_parametric={
            "test": "Kruskal-Wallis H",
            "H": float(h_stat),
            "p_value": float(h_p),
            "df": float(len(groups) - 1),
        },
        effect_size={
            "eta_squared": _eta_squared(groups),
            "epsilon_squared": _epsilon_squared(float(h_stat), n_total, len(groups)),
        },
        posthoc=posthoc_df,
        trend=trend,
        notes=notes,
    )


def correlation(df: pd.DataFrame, metric: str, var: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Correlate two numeric columns and regress ``metric`` on ``var``.

    Parameters
    ----------
    df : pandas.DataFrame
        Contains ``metric`` and ``var``; both are coerced to numbers and rows
        with a missing value are dropped.
    metric, var : str
        Column names.

    Returns
    -------
    result : dict
        ``n``, ``pearson_r``, ``pearson_p``, ``spearman_rho``, ``spearman_p``,
        and the least-squares fit ``metric = slope * var + intercept`` as
        ``slope``, ``intercept``, ``slope_se``, ``regression_p`` and
        ``r_squared``; or ``n`` and ``error`` with fewer than 3 rows.
    data : pandas.DataFrame
        The cleaned two-column frame the statistics were computed on.
    """
    work = pd.DataFrame(
        {
            metric: pd.to_numeric(df[metric], errors="coerce"),
            var: pd.to_numeric(df[var], errors="coerce"),
        }
    ).dropna()
    n = int(work.shape[0])
    if n < MIN_N_FOR_CORRELATION:
        return {"n": n, "error": INSUFFICIENT_DATA}, work

    r_p, p_p = stats.pearsonr(work[metric], work[var])
    r_s, p_s = stats.spearmanr(work[metric], work[var])
    fit = stats.linregress(work[var], work[metric])
    return {
        "n": n,
        "pearson_r": float(r_p),
        "pearson_p": float(p_p),
        "spearman_rho": float(r_s),
        "spearman_p": float(p_s),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "slope_se": float(fit.stderr),
        "regression_p": float(fit.pvalue),
        "r_squared": float(fit.rvalue**2),
    }, work
