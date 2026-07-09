"""Reducing redundancy.

1. Correlated factors  - time-series average of cross-sectional correlations;
                         flag pairs with |rho| above the configured threshold.
2. Clustering          - group correlated factors; represent each cluster by its
                         largest-|Fama-MacBeth-coefficient| member (``fm_gradient``),
                         its highest IC-IR member (``best_ir``), or an
                         equal-weighted z-score composite.
3. Parsimony           - lasso on a predictive regression of forward returns on the
                         supplied neutralized factors; survivors are shortlisted.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl

from ._common import (
    SUBUNIVERSES,
    applicable_factor_columns,
    as_df,
    cross_sectional_residuals,
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


def select_cluster_representatives(
    clusters, ic_ir=None, scores=None, method="best_ir", fm=None
):
    """Represent each cluster by one factor or an equal-weighted z-score composite.

    * ``method="fm_gradient"`` -- keep the factor with the largest absolute
      Fama-MacBeth gradient (|mean coefficient|, the per-unit-z premium) per
      cluster. ``fm`` maps factor -> |coef| (a dict, or the frame from
      :func:`single_factor.fama_macbeth`; with ``pooled=True`` each factor has
      exactly one row, otherwise the max |coef| across sub-universes is used).
      A member without a finite coefficient falls back to its |IC-IR| from
      ``ic_ir``; members with a coefficient always outrank members without one.
      Returns the list of representative factor names.
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
    if method == "fm_gradient":
        grad = _fm_gradient_lookup(fm)
        ir = _ir_lookup(ic_ir)

        def gradient_key(f):
            g = grad.get(f)
            if g is not None and np.isfinite(g):
                return (1, abs(g))
            v = ir.get(f)
            return (0, abs(v) if v is not None and np.isfinite(v) else float("-inf"))

        return [max(cluster, key=gradient_key) for cluster in clusters]

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


def _fm_gradient_lookup(fm) -> dict:
    """Normalise an FM source (dict or fama_macbeth frame) to factor -> |coef|."""
    if fm is None:
        return {}
    if isinstance(fm, dict):
        return fm
    if isinstance(fm, pl.DataFrame):
        best = (
            fm.filter(pl.col("factor") != "const")
            .drop_nulls("mean_coef")
            .filter(pl.col("mean_coef").is_finite())
            .group_by("factor")
            .agg(pl.col("mean_coef").abs().max().alias("grad"))
        )
        return dict(zip(best["factor"].to_list(), best["grad"].to_list()))
    raise TypeError("fm must be a dict or a polars DataFrame.")


def lasso_select(
    fwd_returns,
    neutralized_scores,
    cfg,
    nonstyle_exposures=None,
    target_col="excess_return",
    universe_col="industry",
    *,
    delist_events,
):
    """Lasso predictive regression per sub-universe; return the surviving factors.

    Penalized regression is run **within each sub-universe**. Within a sub-universe the design is
    dense, the regressors are standardized so the L1 penalty is scale-fair, and a
    cross-validated lasso (``LassoCV``) is fit. Factors driven to zero are dropped;
    a factor survives if it survives in *any* sub-universe.

    When ``nonstyle_exposures`` is given, the forward returns are first residualised
    cross-sectionally against the non-style design (per period), exactly as
    :func:`single_factor.rank_ic` does: the neutralized (market-orthogonal) style
    scores are then regressed on a like-for-like target, instead of on raw returns
    whose ranks and variance are dominated by the un-modelled market risk the scores
    were neutralized *against*.

    ``delist_events`` (required keyword) is passed to :func:`forward_returns` so
    terminal delisting returns enter the lasso target (``None`` opts out).

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
    fwd = forward_returns(
        fwd_returns, lags=(1,), target_col=target_col, period_months=period,
        delist_events=delist_events,
    )
    if nonstyle_exposures is not None:
        # Sample each security's loadings at its rebalance date, then residualise the
        # forward return per period (the common cross-section key) -- so the lasso
        # target is net of non-style risk, matching the neutralized regressors.
        exposures = fwd.select("stock_id", "date", "period").join(
            as_df(nonstyle_exposures), on=["stock_id", "date"], how="left"
        )
        fwd = cross_sectional_residuals(fwd, ["_fwd1"], exposures, by="period")
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

    Folds group whole rebalancing periods together so a period's cross-section 
    never straddles the train/test split, preventing cross-sectional leakage.
    """
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import TimeSeriesSplit
    unique_periods = np.sort(np.unique(periods))
    n_splits = min(5, len(unique_periods))
    
    if n_splits < 2:
        return np.zeros(X.shape[1])
        
    # 1. Split on the unique periods, not the raw data
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv = []
    
    for train_period_idx, test_period_idx in tscv.split(unique_periods):
        # 2. Identify the actual periods for this fold
        train_periods = unique_periods[train_period_idx]
        test_periods = unique_periods[test_period_idx]
        
        # 3. Map the periods back to the original row indices in X
        train_idx = np.where(np.isin(periods, train_periods))[0]
        test_idx = np.where(np.isin(periods, test_periods))[0]
        
        cv.append((train_idx, test_idx))

    # 4. Pass the custom cv iterator to LassoCV
    return LassoCV(cv=cv, alphas=50, max_iter=10000).fit(X, y).coef_
