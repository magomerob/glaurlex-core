"""Per-type frequency, position and lexical availability statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

AVAILABILITY_DECAY = 2.3

TYPE_STATISTICS_COLUMNS = (
    "type",
    "rel_freq",
    "tokens",
    "availability",
    "avg_pos",
    "appearance",
    "cum_freq",
)


def type_statistics(
    df: pd.DataFrame, informant_ids: Iterable[object] | None = None
) -> pd.DataFrame:
    r"""Compute frequency, position and lexical availability statistics per type.

    Parameters
    ----------
    df : pandas.DataFrame
        Topic frame with columns ``informant_id``, ``pos`` (0-indexed) and ``type``.
    informant_ids : iterable, optional
        Restrict the computation to these informants. All informants are used
        when omitted.

    Returns
    -------
    pandas.DataFrame
        One row per type with columns ``type``, ``rel_freq`` (share of all
        tokens), ``tokens`` (occurrences), ``availability``, ``avg_pos`` (mean
        0-indexed position), ``appearance`` (share of informants producing the
        type) and ``cum_freq`` (cumulative ``rel_freq``). Rows are sorted by
        ``availability`` in descending order, ties by ``type``.

    Notes
    -----
    With :math:`N` informants, :math:`n` the highest position in the topic and
    :math:`u_{t,i}` the number of informants who produced type :math:`t` at
    position :math:`i`, the availability index is

    .. math::

        D(t) = \sum_{i=0}^{n} e^{-2.3\, i / n} \, \frac{u_{t,i}}{N}

    following López Chávez and Strassburger Frías (1991). When every response
    sits at position 0 (:math:`n = 0`) the index is set to 1 for every type.

    Runs in :math:`O(R \log R)` for :math:`R` responses.
    """
    if informant_ids is not None:
        df = df[df["informant_id"].isin(set(informant_ids))]
    if df.empty:
        return pd.DataFrame(columns=list(TYPE_STATISTICS_COLUMNS))

    n_informants = df["informant_id"].nunique()
    counts = df["type"].value_counts()
    out = pd.DataFrame(
        {
            "type": counts.index,
            "rel_freq": df["type"].value_counts(normalize=True).to_numpy(),
            "tokens": counts.to_numpy(),
        }
    )
    by_type = df.groupby("type")
    appearance = by_type["informant_id"].nunique() / n_informants

    max_pos = int(df["pos"].max())
    if max_pos == 0:
        out["availability"] = 1.0
        out["avg_pos"] = 0.0
    else:
        per_position = (
            df.groupby(["type", "pos"])["informant_id"].nunique().rename("u").reset_index()
        )
        weights = np.exp(-AVAILABILITY_DECAY * per_position["pos"].to_numpy() / max_pos)
        terms = pd.Series(weights * per_position["u"].to_numpy() / n_informants)
        availability = terms.groupby(per_position["type"].to_numpy()).sum()
        out["availability"] = out["type"].map(availability).fillna(0.0).to_numpy()
        out["avg_pos"] = out["type"].map(by_type["pos"].mean()).to_numpy()
    out["appearance"] = out["type"].map(appearance).to_numpy()

    out = out.sort_values(["availability", "type"], ascending=[False, True])
    out["cum_freq"] = out["rel_freq"].cumsum()
    return out
