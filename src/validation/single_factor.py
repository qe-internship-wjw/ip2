"""Single-factor evaluation.

1. Information Coefficient
   - Rank IC between the exposure and the forward return. When non-style exposures
     are supplied the forward return is first residualised against them, so the
     IC measures the part of the return *not explained by non-style factors*.
   - IC decay across forward horizons to inform rebalancing frequency / turnover.
   - Information ratio of the IC series; consistent IR > 0.3 shortlists a factor.
2. Quantile return profiling
   - Cross-sectionally bucket the exposure into quantiles each period, simulate the
     forward return of each equal-weighted quantile portfolio, and average over
     periods. The shape of the profile exposes non-linearity that a single Rank IC
     (a linear rank measure) would hide.
3. Fama-MacBeth
   - Per-period cross-sectional regressions of forward returns on scores, split by
     sub-universe, aggregated with Newey-West t-stats for autocorrelation.

Cross-sectional regressions run as vectorised ``polars-ols`` expressions
``.over("date")``; the Newey-West (HAC) aggregation uses ``statsmodels``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import polars_ols  # noqa: F401 -- registers the `.least_squares` expr namespace
import statsmodels.api as sm

from ._common import (
    SUBUNIVERSES,
    applicable_factor_columns,
    as_df,
    cross_sectional_residuals,
    factor_columns,
    forward_returns,
    subuniverse_mask,
)


def rank_ic(
    scores,
    fwd_returns,
    lags=(1, 2, 3),
    nonstyle_exposures=None,
    target_col="excess_return",
    period_months=3,
    winsorize_limits=(0.01, 0.99),
):
    """Rank IC series and IC decay across the given forward lags.

    For each factor and each lag, computes the per-date Spearman rank correlation
    between the time-``t`` exposure and the return realised ``lag`` *rebalancing
    periods* ahead (see :func:`forward_returns`). When ``nonstyle_exposures`` is
    given, the forward returns are first residualised cross-sectionally against
    those factors (per date), so the IC is *net of non-style risk* as specified in
    the Experiment Plan.

    A sector factor is null outside its sub-universe, so its per-period IC is
    formed only over the sub-universe where it is defined (the ``drop_nulls``
    below). Cross-sections are grouped on the common ``period`` key (securities'
    period-end days differ under staggered trading calendars).

    Returns a long frame ``[period, factor, lag, ic]`` -- the IC time series whose
    behaviour across lags is the IC-decay curve.
    """
    scores = as_df(scores)
    fac_cols = factor_columns(scores)
    fwd = forward_returns(
        fwd_returns, lags=lags, target_col=target_col,
        period_months=period_months, winsorize_limits=winsorize_limits,
    )

    if nonstyle_exposures is not None:
        # Sample each security's loadings at its rebalance date, then residualise
        # the forward returns per period (the common cross-section key).
        exposures = fwd.select("stock_id", "date", "period").join(
            as_df(nonstyle_exposures), on=["stock_id", "date"], how="left"
        )
        fwd = cross_sectional_residuals(
            fwd, [f"_fwd{lag}" for lag in lags], exposures, by="period"
        )

    df = scores.join(fwd, on=["stock_id", "date"], how="inner")

    frames = []
    for lag in lags:
        fwd_col = f"_fwd{lag}"
        for f in fac_cols:
            frames.append(
                df.drop_nulls([f, fwd_col])
                .group_by("period")
                .agg(ic=pl.corr(f, fwd_col, method="spearman"))
                .with_columns(factor=pl.lit(f), lag=pl.lit(lag))
            )

    # A degenerate cross-section (constant factor / <2 points) yields NaN from
    # pl.corr; map it to null so downstream mean/std/IR treat it as missing.
    return (
        pl.concat(frames)
        .select("period", "factor", "lag", pl.col("ic").fill_nan(None))
        .sort("factor", "lag", "period")
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


def quantile_returns(
    scores,
    fwd_returns,
    n_quantiles=10,
    lag=1,
    nonstyle_exposures=None,
    target_col="excess_return",
    period_months=3,
    winsorize_limits=(0.01, 0.99),
    weight_col=None,
):
    """Forward-return profile of quantile portfolios, one per factor.

    Buckets are formed per period as in :func:`rank_ic`; daily returns are winsorized
    before compounding (``winsorize_limits``, see :func:`forward_returns`) so no extreme
    print drives a bucket's weighted mean.

    Each bucket's portfolio return is **equal-weighted** by default. Pass
    ``weight_col`` (e.g. ``"mcap_usd"``) to cap-weight instead: the column is sampled
    at the formation date (:func:`forward_returns`) and each bucket return becomes
    ``sum(w * ret) / sum(w)`` over its members. ``weight_col`` must be present in the
    ``fwd_returns`` frame (e.g. ``sector_panel.select("stock_id", "date",
    "excess_return", "mcap_usd")``); a bucket whose weights sum to zero yields null.

    Returns a long frame ``[factor, quantile, mean_ret, std_ret, n_periods]`` where
    ``quantile`` runs ``1`` (lowest exposure) .. ``n_quantiles`` (highest) and
    ``mean_ret`` is the average per-period return of that quantile portfolio.
    """
    scores = as_df(scores)
    fac_cols = factor_columns(scores)
    fwd = forward_returns(
        fwd_returns, lags=(lag,), target_col=target_col,
        period_months=period_months, winsorize_limits=winsorize_limits,
        weight_col=weight_col,
    )
    fwd_col = f"_fwd{lag}"

    if nonstyle_exposures is not None:
        # Sample each security's loadings at its rebalance date, then residualise
        # the forward returns per period (the common cross-section key). The weight
        # column is a non-target passthrough, so it survives the residualization.
        exposures = fwd.select("stock_id", "date", "period").join(
            as_df(nonstyle_exposures), on=["stock_id", "date"], how="left"
        )
        fwd = cross_sectional_residuals(fwd, [fwd_col], exposures, by="period")

    df = scores.join(fwd, on=["stock_id", "date"], how="inner")

    # rank in [1, N] over the period -> integer bucket in [1, n_quantiles].
    bucket = ((pl.col("_rank") - 1) * n_quantiles) // pl.len().over("period") + 1

    # Per (period, bucket) portfolio return: equal- or (mcap-)weighted mean.
    if weight_col is not None:
        wsum = pl.col(weight_col).sum()
        port_ret = (
            pl.when(wsum > 0)
            .then((pl.col(fwd_col) * pl.col(weight_col)).sum() / wsum)
            .otherwise(None)
        )
    else:
        port_ret = pl.col(fwd_col).mean()

    frames = []
    for f in fac_cols:
        drop_cols = [f, fwd_col] + ([weight_col] if weight_col is not None else [])
        valid = df.drop_nulls(drop_cols).with_columns(
            _rank=pl.col(f).rank(method="ordinal").over("period").cast(pl.Int64)
        )
        frames.append(
            valid.with_columns(quantile=bucket)
            # weight the members within each period, then average across periods.
            .group_by("period", "quantile")
            .agg(port_ret=port_ret)
            .group_by("quantile")
            .agg(
                mean_ret=pl.col("port_ret").mean(),
                std_ret=pl.col("port_ret").std(),
                n_periods=pl.len(),
            )
            .with_columns(factor=pl.lit(f))
        )

    return (
        pl.concat(frames)
        .select("factor", "quantile", "mean_ret", "std_ret", "n_periods")
        .sort("factor", "quantile")
    )


def _hac_lags(n: int, override) -> int:
    """Newey-West maxlags: caller override, else the ``4 (T/100)^(2/9)`` rule."""
    lags = override if override is not None else max(1, int(4 * (n / 100) ** (2 / 9)))
    return min(lags, n - 1)

_FM_SCHEMA = {
    "sub_universe": pl.Utf8, "factor": pl.Utf8, "mean_coef": pl.Float64,
    "t_stat": pl.Float64, "nw_se": pl.Float64, "n_periods": pl.Int64,
}


def fama_macbeth(
    scores,
    fwd_returns,
    newey_west_lags=None,
    target_col="excess_return",
    period_months=3,
    universe_col="industry",
    winsorize_limits=(0.01, 0.99),
):
    """Aggregate cross-sectional regression premia with Newey-West t-stats.

    Regression is run **per sub-universe**: Within a sub-universe the design is dense,
    so the usual multivariate Fama-MacBeth applies (a per-period cross-sectional
    OLS, then a Newey-West time-series aggregation of the coefficient series).

    Returns a frame ``[sub_universe, factor, mean_coef, t_stat, nw_se,
    n_periods]`` (``factor`` includes ``const``); all-financials factors appear
    once per sub-universe.
    """
    scores = as_df(scores)
    if universe_col not in scores.columns:
        raise ValueError(
            f"fama_macbeth needs the '{universe_col}' sub-universe label to split "
            "Banks vs Insurers; otherwise the mutually-exclusive sector factors "
            "drop every row. Attach it via universe.industry_labels."
        )
    fwd = forward_returns(
        fwd_returns, lags=(1,), target_col=target_col,
        period_months=period_months, winsorize_limits=winsorize_limits,
    )
    df = scores.join(fwd.rename({"_fwd1": "_y"}), on=["stock_id", "date"], how="inner")

    frames = []
    for sub, applic in SUBUNIVERSES.items():
        fac_cols = applicable_factor_columns(scores, applic)
        # Within a sub-universe the applicable factors form a dense design; drop
        # only rows still carrying a null (e.g. a loading's warm-up window).
        sub_df = df.filter(subuniverse_mask(sub, universe_col)).drop_nulls(["_y", *fac_cols])
        premia = _fama_macbeth_premia(sub_df, fac_cols, newey_west_lags)
        frames.append(premia.with_columns(sub_universe=pl.lit(sub)))

    return pl.concat(frames, how="vertical").select(*_FM_SCHEMA)


def _fama_macbeth_premia(df, fac_cols, newey_west_lags):
    """Per-period cross-sectional OLS + Newey-West aggregation for one dense panel.

    Returns ``[factor, mean_coef, t_stat, nw_se, n_periods]`` with ``factor`` in
    ``["const", *fac_cols]``.
    """
    names = ["const", *fac_cols]
    if df.height == 0 or not fac_cols:
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
        .over("period")
    )
    per_period = df.select("period", _coef=coef).unique(subset="period", keep="first").sort("period")

    rows = []
    for name in names:
        series = per_period["_coef"].struct.field(name).to_numpy()
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
