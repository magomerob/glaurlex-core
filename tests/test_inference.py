import numpy as np
import pandas as pd
import pytest
from helpers import assert_matches
from scipy import stats

from glaurlex.core.inference import (
    compare_groups,
    correlation,
    describe_series,
    informant_metrics,
    nonparametric_p_value,
    normality_test,
    qq_points,
)
from glaurlex.core.stats import type_statistics


@pytest.fixture(scope="module")
def metrics(topic, informants):
    return informant_metrics(topic, informants)


def test_informant_metrics_parity(metrics, reference):
    assert_matches(metrics, reference["informant_metrics"])


def test_informant_metrics_on_a_hand_computed_list():
    df = pd.DataFrame(
        {"informant_id": [7, 7, 7, 8], "pos": [0, 1, 2, 0], "type": ["a", "b", "a", "c"]}
    )
    informants = pd.DataFrame({"informant_id": [7, 8, 9], "SEX": ["F", "M", "F"]})
    availability = type_statistics(df).set_index("type")["availability"]
    row = informant_metrics(df, informants).set_index("informant_id").loc[7]

    assert row[["n_tokens", "n_types", "max_pos", "SEX"]].tolist() == [3, 2, 3, "F"]
    assert row["mean_pos"] == pytest.approx(2.0)
    assert row["ttr"] == pytest.approx(2 / 3)
    assert row["type_coverage"] == pytest.approx(2 / 3)
    assert row["entropy"] == pytest.approx(stats.entropy([2, 1], base=2))
    assert row["total_availability"] == pytest.approx(2 * availability["a"] + availability["b"])


def test_informants_with_the_same_responses_tie_exactly():
    lists = {1: "dhag", 2: "dh", 3: "dgha", 4: "aabbbbcccddd", 5: "pppqqqrrrrss"}
    df = pd.DataFrame(
        [(i, pos, t) for i, words in lists.items() for pos, t in enumerate(words)],
        columns=["informant_id", "pos", "type"],
    )
    out = informant_metrics(df, pd.DataFrame({"informant_id": list(lists)})).set_index(
        "informant_id"
    )
    columns = ["total_availability", "mean_availability", "entropy"]

    assert out.loc[1, columns].tolist() == out.loc[3, columns].tolist()
    assert out.loc[4, "entropy"] == out.loc[5, "entropy"]


def test_informant_metrics_excludes_absent_informants(metrics, informants):
    assert set(metrics["informant_id"]) <= set(informants["informant_id"])
    assert metrics["informant_id"].is_monotonic_increasing


def test_informant_metrics_requires_informant_id(topic):
    with pytest.raises(ValueError, match="informant_id"):
        informant_metrics(topic, pd.DataFrame({"CODE": [1]}))


def test_describe_and_normality_parity(metrics, reference):
    values = metrics["mean_availability"]

    assert_matches(describe_series(values), reference["describe_mean_availability"])
    result = normality_test(values)
    assert result.pop("test") == "Shapiro-Wilk"
    assert_matches(result, reference["normality_mean_availability"])


def test_describe_degenerate_inputs():
    assert describe_series(pd.Series(["x", None])) == {"n": 0}
    single = describe_series(pd.Series([4.0]))
    assert single["std"] == 0.0
    assert np.isnan(single["skewness"])
    assert np.isnan(single["kurtosis"])


def test_normality_switches_test_by_sample_size():
    rng = np.random.default_rng(0)
    assert normality_test(pd.Series([1.0, 2.0]))["error"] == "insufficient_data"
    large = normality_test(pd.Series(rng.normal(size=5001)))
    assert large["test"] == "D'Agostino-Pearson"
    assert large["is_normal"]
    assert not normality_test(pd.Series(rng.exponential(size=200)))["is_normal"]


def _result_payload(result):
    return {
        "kind": result.kind,
        "n_total": result.n_total,
        "descriptives": result.descriptives,
        "parametric": {k: v for k, v in result.parametric.items() if k != "test"},
        "non_parametric": {k: v for k, v in result.non_parametric.items() if k != "test"},
        "effect_size": result.effect_size,
        "trend": result.trend,
        "posthoc": result.posthoc,
    }


def test_two_group_comparison_parity(metrics, reference):
    result = compare_groups(metrics, "n_tokens", "SEX")

    assert result.parametric["test"] == "Welch's t-test"
    assert_matches(_result_payload(result), reference["compare_n_tokens_by_sex"])


def test_ordinal_k_group_comparison_parity(metrics, reference):
    result = compare_groups(metrics, "n_types", "LEVEL", order=["C", "A", "B"])

    assert result.descriptives["LEVEL"].tolist() == ["C", "A", "B"]
    assert result.posthoc[["group_1", "group_2"]].to_numpy().tolist() == [
        ["C", "A"],
        ["C", "B"],
        ["A", "B"],
    ]
    assert_matches(_result_payload(result), reference["compare_n_types_by_level"])


def test_posthoc_pairs_match_the_tested_levels():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "g": ["solo"] + ["x"] * 6 + ["y"] * 6 + ["z"] * 6,
            "v": [100.0, *rng.normal(0, 1, 6), *rng.normal(3, 1, 6), *rng.normal(6, 1, 6)],
        }
    )
    result = compare_groups(df, "v", "g")

    assert result.kind == "k-group"
    assert result.n_total == 18
    pairs = result.posthoc.set_index(["group_1", "group_2"])
    assert list(pairs.index) == [("x", "y"), ("x", "z"), ("y", "z")]
    for (a, b), row in pairs.iterrows():
        u, p = stats.mannwhitneyu(df.v[df.g == a], df.v[df.g == b], alternative="two-sided")
        assert row["U"] == u
        assert row["p_bonferroni"] == pytest.approx(min(3 * p, 1.0))


def test_order_with_unknown_and_missing_levels():
    df = pd.DataFrame({"g": list("aabbcc"), "v": [1, 2, 3, 4, 5, 6]})
    result = compare_groups(df, "v", "g", order=["c", "zzz", "a"])

    assert result.descriptives["g"].tolist() == ["c", "a", "b"]
    assert result.trend["levels"] == ["c", "a", "b"]
    assert result.trend["spearman_rho"] < 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_insufficient_and_constant_groups():
    insufficient = compare_groups(pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]}), "v", "g")
    assert insufficient.kind == "insufficient"
    assert insufficient.notes == ["insufficient_levels"]

    constant_two = compare_groups(pd.DataFrame({"g": list("aabb"), "v": [1.0] * 4}), "v", "g")
    assert constant_two.notes == ["welch_t_undefined"]

    constant_k = compare_groups(
        pd.DataFrame({"g": list("aabbcc"), "v": [1.0] * 6}), "v", "g", posthoc=False
    )
    assert constant_k.notes == ["anova_undefined", "kruskal_undefined"]
    assert constant_k.posthoc is None


def test_correlation_parity_and_exact_fit(metrics, reference):
    result, data = correlation(metrics, "n_tokens", "AGE")
    assert_matches(result, reference["correlation_n_tokens_age"])
    assert len(data) == len(metrics)

    x = np.arange(10.0)
    exact, _ = correlation(pd.DataFrame({"y": 2 * x + 1, "x": x}), "y", "x")
    assert (exact["slope"], exact["intercept"], exact["r_squared"]) == pytest.approx((2, 1, 1))

    short, _ = correlation(pd.DataFrame({"y": [1, None, 3], "x": [1, 2, "?"]}), "y", "x")
    assert short == {"n": 1, "error": "insufficient_data"}


def test_nonparametric_p_value_dispatch():
    a, b, c = np.array([1.0, 2, 3]), np.array([4.0, 5, 6]), np.array([7.0, 8, 9])

    assert nonparametric_p_value([a, b, np.array([1.0])]) == stats.mannwhitneyu(a, b).pvalue
    assert nonparametric_p_value([a, b, c]) == stats.kruskal(a, b, c).pvalue
    assert np.isnan(nonparametric_p_value([a, np.array([1.0])]))


def test_qq_points_match_probplot():
    values = pd.Series([3.0, 1.0, "x", 2.0, 10.0])
    qq = qq_points(values)
    (theoretical, _), (slope, intercept, r) = stats.probplot([3.0, 1.0, 2.0, 10.0])

    np.testing.assert_allclose(qq.theoretical, theoretical)
    np.testing.assert_array_equal(qq.observed, [1.0, 2.0, 3.0, 10.0])
    assert (qq.slope, qq.intercept, qq.r) == pytest.approx((slope, intercept, r))
