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

from ._common import as_df, factor_columns, forward_returns


def average_correlation(scores, threshold=0.6):
    """Mean cross-sectional factor correlation; flag pairs above threshold.

    For every factor pair, the Pearson correlation is computed within each date's
    cross-section and averaged over dates (a time-series average of cross-sectional
    correlations, so no cross-date leakage).

    Returns ``(pairs, flagged)``: ``pairs`` is ``[factor_a, factor_b, rho]`` for
    all pairs; ``flagged`` is the subset with ``|rho| > threshold``, most
    correlated first.
    """
    df = as_df(scores)
    cols = factor_columns(df)

    records = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            per_date = df.drop_nulls([a, b]).group_by("date").agg(
                rho=pl.corr(a, b)
            )
            records.append(
                {"factor_a": a, "factor_b": b,
                 "rho": per_date["rho"].drop_nulls().mean()}
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


def lasso_select(fwd_returns, neutralized_scores, cfg):
    """Elastic-net/lasso predictive regression; return surviving factors.

    Pools all (stock, date) observations into one predictive regression of the
    next-period return on every neutralized factor, standardizes the regressors so
    the L1 penalty is scale-fair, and fits a cross-validated lasso. Factors whose
    coefficient is driven to zero are dropped; the rest are the shortlist.

    Uses scikit-learn's ``LassoCV`` when available, falling back to a
    cross-validated statsmodels elastic net (``L1_wt=1``) otherwise.
    """
    scores = as_df(neutralized_scores)
    cols = factor_columns(scores)
    fwd, _ = forward_returns(fwd_returns, lags=(1,))
    df = scores.join(
        fwd.rename({"_fwd1": "_y"}), on=["stock_id", "date"], how="inner"
    ).drop_nulls(["_y", *cols])

    X = df.select(cols).to_numpy()
    y = df["_y"].to_numpy().astype(float)

    # Scale-fair penalty: standardize regressors (drop degenerate columns).
    mu, sd = X.mean(axis=0), X.std(axis=0)
    keep = sd > 0
    Xz = np.zeros_like(X)
    Xz[:, keep] = (X[:, keep] - mu[keep]) / sd[keep]

    coef = _lasso_coefficients(Xz, y - y.mean(), cfg)
    return [c for c, b, k in zip(cols, coef, keep) if k and abs(b) > 0.0]


def _lasso_coefficients(X: np.ndarray, y: np.ndarray, cfg) -> np.ndarray:
    """Fitted lasso coefficients (sklearn if present, else statsmodels CV)."""
    try:
        from sklearn.linear_model import LassoCV

        return LassoCV(cv=5, n_alphas=50, max_iter=10000).fit(X, y).coef_
    except ImportError:
        import statsmodels.api as sm

        n = X.shape[0]
        rng = np.arange(n)
        folds = np.array_split(rng, 5)
        alphas = np.logspace(-4, 0, 25)

        best_alpha, best_mse = alphas[0], np.inf
        for alpha in alphas:
            errs = []
            for k in range(5):
                test = folds[k]
                train = np.setdiff1d(rng, test)
                params = (
                    sm.OLS(y[train], X[train])
                    .fit_regularized(alpha=alpha, L1_wt=1.0)
                    .params
                )
                errs.append(np.mean((y[test] - X[test] @ params) ** 2))
            mse = float(np.mean(errs))
            if mse < best_mse:
                best_alpha, best_mse = alpha, mse

        return np.asarray(
            sm.OLS(y, X).fit_regularized(alpha=best_alpha, L1_wt=1.0).params
        )
