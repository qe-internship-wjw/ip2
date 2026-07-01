"""Single-factor evaluation.

1. Information Coefficient
   - Rank IC between t exposure and t+1 excess return. When non-style exposures
     are supplied the forward return is first residualised against them, so the
     IC measures the part of the return *not explained by non-style factors*.
   - IC decay at t+1, t+2, t+3 to inform rebalancing frequency / turnover.
   - Information ratio of the IC series; consistent IR > 0.3 shortlists a factor.
2. Quantile Portfolio
   - Sort the universe into quantiles; equal- or cap-weighted forward returns.
   - Long-Short (Q1-Qn) returns with Newey-West t-stats and Sharpe.
3. Fama-MacBeth
   - Period-by-period cross-sectional regressions of forward returns on scores.
   - Aggregate coefficients with Newey-West adjusted t-stats for autocorrelation.

Cross-sectional regressions (IC residualisation, Fama-MacBeth) run as vectorised
``polars-ols`` expressions ``.over("date")``; the time-series Newey-West (HAC)
aggregation uses ``statsmodels`` (polars-ols has no HAC covariance).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import polars_ols  # noqa: F401 -- registers the `.least_squares` expr namespace
import statsmodels.api as sm

from ._common import as_df, cross_sectional_residuals, factor_columns, forward_returns


def rank_ic(scores, fwd_returns, lags=(1, 2, 3), nonstyle_exposures=None):
    """Rank IC series and IC decay across the given forward lags.

    For each factor and each lag, computes the per-date Spearman rank correlation
    between the time-``t`` exposure and the return realised ``lag`` periods ahead.
    When ``nonstyle_exposures`` is given, the forward returns are first
    residualised cross-sectionally against those factors (per date), so the IC is
    *net of non-style risk* as specified in the Experiment Plan.

    Returns a long frame ``[date, factor, lag, ic]`` -- the IC time series whose
    behaviour across lags is the IC-decay curve.
    """
    scores = as_df(scores)
    fac_cols = factor_columns(scores)
    fwd, _ = forward_returns(fwd_returns, lags=lags)

    if nonstyle_exposures is not None:
        fwd = cross_sectional_residuals(
            fwd, [f"_fwd{lag}" for lag in lags], nonstyle_exposures, by="date"
        )

    df = scores.join(fwd, on=["stock_id", "date"], how="inner")

    frames = []
    for lag in lags:
        fwd_col = f"_fwd{lag}"
        for f in fac_cols:
            frames.append(
                df.drop_nulls([f, fwd_col])
                .group_by("date")
                .agg(ic=pl.corr(f, fwd_col, method="spearman"))
                .with_columns(factor=pl.lit(f), lag=pl.lit(lag))
            )

    return pl.concat(frames).select("date", "factor", "lag", "ic").sort(
        "factor", "lag", "date"
    )


def information_ratio(ic_series):
    """IR = mean(IC) / std(IC) of the IC time series.

    Accepts a raw series/iterable of ICs (returns a float) or the long frame from
    :func:`rank_ic` (returns per ``factor``/``lag`` IRs as a frame).
    """
    if isinstance(ic_series, pl.DataFrame):
        keys = [c for c in ("factor", "lag") if c in ic_series.columns]
        ic = pl.col("ic")
        agg = pl.when(ic.std() > 0).then(ic.mean() / ic.std()).otherwise(None).alias("ir")
        if keys:
            return ic_series.group_by(keys).agg(agg).sort(keys)
        ic_series = ic_series["ic"]

    s = ic_series if isinstance(ic_series, pl.Series) else pl.Series(list(ic_series))
    s = s.drop_nulls()
    std = s.std()
    if std is None or std == 0:
        return float("nan")
    return float(s.mean() / std)


def _hac_lags(n: int, override) -> int:
    """Newey-West maxlags: caller override, else the ``4 (T/100)^(2/9)`` rule."""
    lags = override if override is not None else max(1, int(4 * (n / 100) ** (2 / 9)))
    return min(lags, n - 1)

def fama_macbeth(scores, fwd_returns, newey_west_lags=None):
    """Aggregate cross-sectional regression premia with Newey-West t-stats.

    Returns a frame ``[factor, mean_coef, t_stat, nw_se, n_periods]`` (``factor``
    includes ``const``).
    """
    scores = as_df(scores)
    fac_cols = factor_columns(scores)
    fwd, _ = forward_returns(fwd_returns, lags=(1,))
    df = scores.join(fwd.rename({"_fwd1": "_y"}), on=["stock_id", "date"], how="inner")
    df = df.drop_nulls(["_y", *fac_cols])

    names = ["const", *fac_cols]
    if df.height == 0:
        return pl.DataFrame(
            {"factor": names, "mean_coef": [np.nan] * len(names),
             "t_stat": [np.nan] * len(names), "nw_se": [np.nan] * len(names),
             "n_periods": [0] * len(names)}
        )

    coef = (
        pl.col("_y")
        .least_squares.ols(
            *fac_cols, mode="coefficients", add_intercept=True,
            null_policy="drop", solve_method="svd",
        )
        .over("date")
    )
    per_date = df.select("date", _coef=coef).unique(subset="date", keep="first").sort("date")

    rows = []
    for name in names:
        series = per_date["_coef"].struct.field(name).to_numpy()
        series = series[np.isfinite(series)]
        periods = series.size
        if periods < 2:
            rows.append({"factor": name, "mean_coef": float(series.mean()) if periods else np.nan,
                         "t_stat": np.nan, "nw_se": np.nan, "n_periods": periods})
            continue
        fit = sm.OLS(series, np.ones(periods)).fit(
            cov_type="HAC", cov_kwds={"maxlags": _hac_lags(periods, newey_west_lags)}
        )
        rows.append(
            {"factor": name, "mean_coef": float(series.mean()),
             "t_stat": float(fit.tvalues[0]), "nw_se": float(fit.bse[0]),
             "n_periods": periods}
        )
    return pl.DataFrame(rows)
