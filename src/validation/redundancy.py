"""Reducing redundancy.

1. Correlated factors  - time-series average of cross-sectional correlations;
                         flag pairs with rho > 0.6.
2. Clustering          - group correlated factors; represent each cluster by its
                         highest IC-IR factor or an equal-weighted z-score composite.
3. Parsimony           - lasso on a predictive regression of forward returns on all
                         neutralized factors; survivors are shortlisted.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl

from ._common import (
    SUBUNIVERSES,
    applicable_factor_columns,
    as_df,
    factor_columns,
    forward_returns,
    subuniverse_mask,
)


def average_correlation(scores, threshold=0.6):
    """Mean cross-sectional factor correlation; flag pairs above threshold.

    For every factor pair, the Pearson correlation is computed within each
    cross-section and averaged over time (a time-series average of cross-sectional
    correlations, so no cross-period leakage). The cross-section key is ``period``
    when present (one cross-section per rebalance -- so staggered period-end dates
    are not split), else ``date``.

    Returns ``(pairs, flagged)``: ``pairs`` is ``[factor_a, factor_b, rho]`` for
    all pairs; ``flagged`` is the subset with ``|rho| > threshold``, most
    correlated first.
    """
    df = as_df(scores)
    cols = factor_columns(df)
    by = "period" if "period" in df.columns else "date"

    records = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            per_xs = df.drop_nulls([a, b]).group_by(by).agg(rho=pl.corr(a, b))
            # Drop NaN (degenerate single-member cross-sections) before averaging;
            # otherwise a NaN poisons the mean and slips through the flag filter.
            records.append(
                {"factor_a": a, "factor_b": b,
                 "rho": per_xs["rho"].fill_nan(None).drop_nulls().mean()}
            )

    pairs = pl.DataFrame(
        records, schema={"factor_a": pl.Utf8, "factor_b": pl.Utf8, "rho": pl.Float64}
    )
    flagged = pairs.filter(pl.col("rho").abs() > threshold).sort(
        pl.col("rho").abs(), descending=True
    )
    return pairs, flagged


def cluster_factors(corr, threshold=0.6):
    """Group highly correlated factors into clusters (connected components).

    Accepts the output of :func:`average_correlation` (``(pairs, flagged)`` tuple)
    or a ``[factor_a, factor_b, rho]`` frame. Factors linked by ``|rho| >
    threshold`` land in the same cluster; uncorrelated factors form singletons.

    Returns a list of clusters (each a sorted list of factor names).
    """
    if isinstance(corr, tuple):
        pairs, flagged = corr
    else:
        pairs = corr
        flagged = pairs.filter(pl.col("rho").abs() > threshold)

    nodes = sorted(set(pairs["factor_a"].to_list()) | set(pairs["factor_b"].to_list()))
    parent = {x: x for x in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for a, b in zip(flagged["factor_a"].to_list(), flagged["factor_b"].to_list()):
        union(a, b)

    groups = defaultdict(list)
    for node in nodes:
        groups[find(node)].append(node)
    return [sorted(members) for members in groups.values()]


def select_cluster_representatives(clusters, ic_ir=None, scores=None, method="best_ir"):
    """Represent each cluster by one factor or an equal-weighted z-score composite.

    * ``method="best_ir"`` -- keep the highest IC-IR factor per cluster. ``ic_ir``
      maps factor -> IR (a dict, or the frame from
      :func:`single_factor.information_ratio`). Returns the list of representative
      factor names.
    * ``method="composite"`` -- build one composite per cluster as the equal-
      weighted mean of its members' (already standardized) z-scores, re-standardized
      per date. Requires ``scores``; returns a ``[stock_id, date, <composite>]``
      frame (a singleton cluster keeps its own name; otherwise members are joined
      with ``+``).
    """
    if method == "composite":
        if scores is None:
            raise ValueError("method='composite' requires the `scores` frame.")
        df = as_df(scores)
        composites = []
        for cluster in clusters:
            name = cluster[0] if len(cluster) == 1 else "+".join(cluster)
            mean = pl.mean_horizontal([pl.col(c) for c in cluster])
            m = mean.mean().over("date")
            s = mean.std().over("date")
            composites.append(
                pl.when(s.is_null() | (s == 0)).then(pl.lit(0.0))
                .otherwise((mean - m) / s).alias(name)
            )
        return df.select("stock_id", "date", *composites)

    # method == "best_ir": pick the top-IR member of each cluster.
    ir = _ir_lookup(ic_ir)
    representatives = []
    for cluster in clusters:
        representatives.append(
            max(cluster, key=lambda f: ir.get(f, float("-inf")))
        )
    return representatives


def _ir_lookup(ic_ir) -> dict:
    """Normalise an IR source (dict or information_ratio frame) to factor -> IR."""
    if ic_ir is None:
        return {}
    if isinstance(ic_ir, dict):
        return ic_ir
    if isinstance(ic_ir, pl.DataFrame):
        # From information_ratio: has a 'factor' key and an 'ir' column; if IR is
        # reported per lag, keep the best across lags.
        best = ic_ir.group_by("factor").agg(pl.col("ir").max())
        return dict(zip(best["factor"].to_list(), best["ir"].to_list()))
    raise TypeError("ic_ir must be a dict or a polars DataFrame.")


def lasso_select(
    fwd_returns,
    neutralized_scores,
    cfg,
    target_col="excess_return",
    universe_col="industry",
):
    """Lasso predictive regression per sub-universe; return the surviving factors.

    Penalized regression is run **within each sub-universe**. Within a sub-universe the design is
    dense, the regressors are standardized so the L1 penalty is scale-fair, and a
    cross-validated lasso (``LassoCV``) is fit. Factors driven to zero are dropped;
    a factor survives if it survives in *any* sub-universe.

    Returns the shortlist as a de-duplicated list of factor shorthands.
    """
    scores = as_df(neutralized_scores)
    if universe_col not in scores.columns:
        raise ValueError(
            f"lasso_select needs the '{universe_col}' sub-universe label to split "
            "Banks vs Insurers; otherwise the disjoint sector factors leave no "
            "dense rows. Attach it via universe.industry_labels."
        )
    period = int(cfg.get("backtest", {}).get("rebalancing_frequency_months", 3))
    fwd = forward_returns(fwd_returns, lags=(1,), target_col=target_col, period_months=period)
    df = scores.join(fwd.rename({"_fwd1": "_y"}), on=["stock_id", "date"], how="inner")

    survivors: list[str] = []
    for sub, applic in SUBUNIVERSES.items():
        cols = applicable_factor_columns(scores, applic)
        sub_df = df.filter(subuniverse_mask(sub, universe_col)).drop_nulls(["_y", *cols])
        survivors.extend(_lasso_survivors(sub_df, cols))

    # A factor may survive in one or both sub-universes; keep first-seen order.
    return list(dict.fromkeys(survivors))


def _lasso_survivors(df, cols):
    """Standardized cross-validated lasso on one dense sub-universe panel."""
    if df.height == 0 or not cols:
        return []
    X = df.select(cols).to_numpy()
    y = df["_y"].to_numpy().astype(float)

    # Scale-fair penalty: standardize regressors (drop degenerate columns).
    mu, sd = X.mean(axis=0), X.std(axis=0)
    keep = sd > 0
    Xz = np.zeros_like(X)
    Xz[:, keep] = (X[:, keep] - mu[keep]) / sd[keep]

    coef = _lasso_coefficients(Xz, y - y.mean(), df["period"].to_numpy())
    return [c for c, b, k in zip(cols, coef, keep) if k and abs(b) > 1e-5]


def _lasso_coefficients(X: np.ndarray, y: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Cross-validated lasso coefficients, cross-validating by period.

    Folds group whole rebalancing periods together (``GroupKFold`` on ``periods``)
    so a period's cross-section never straddles the train/test split -- otherwise
    contemporaneous rows leak and the alpha selection is over-optimistic.
    """
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import GroupKFold

    n_splits = min(5, len(np.unique(periods)))
    if n_splits < 2:
        return np.zeros(X.shape[1])
    cv = list(GroupKFold(n_splits=n_splits).split(X, y, periods))
    return LassoCV(cv=cv, alphas=50, max_iter=10000).fit(X, y).coef_
