"""Single-factor evaluation.

1. Information Coefficient
   - Rank IC between t exposure and t+1 excess return (net of non-style factors).
   - IC decay at t+1, t+2, t+3 to inform rebalancing frequency / turnover.
   - Information ratio of the IC series; consistent IR > 0.3 shortlists a factor.
2. Quantile Portfolio
   - Sort universe into quintiles; equal- or cap-weighted forward returns.
   - Monotonicity check and Long-Short (Q1-Q5) returns, t-stats, Sharpe.
3. Fama-MacBeth
   - Period-by-period cross-sectional regressions of forward returns on scores.
   - Aggregate coefficients with Newey-West adjusted t-stats for autocorrelation.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ._common import as_df, factor_columns, forward_returns


def rank_ic(scores, fwd_returns, lags=(1, 2, 3)):
    """Rank IC series and IC decay across the given forward lags.

    For each factor and each lag, computes the per-date Spearman rank correlation
    between the time-``t`` exposure and the return realised ``lag`` periods ahead.

    Returns a long frame ``[date, factor, lag, ic]`` -- the IC time series whose
    behaviour across lags is the IC-decay curve.
    """
    scores = as_df(scores)
    fac_cols = factor_columns(scores)
    fwd, _ = forward_returns(fwd_returns, lags=lags)
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


def _quantile_expr(col: str, n: int) -> pl.Expr:
    """Assign 1..n balanced quantiles per date by rank (1 = lowest score)."""
    count = pl.len().over("date").cast(pl.Int64)
    rank = pl.col(col).rank(method="ordinal").over("date").cast(pl.Int64)
    return (((rank - 1) * n) // count + 1).alias("quantile")


def quantile_portfolios(scores, fwd_returns, n=5, weighting="equal"):
    """Forward returns of ``n`` factor-sorted portfolios, per date and factor.

    At each date the cross-section is sorted on the factor and split into ``n``
    equal-count buckets (quantile 1 = lowest score, ``n`` = highest); the bucket's
    next-period return is equal-weighted, or cap-weighted when ``weighting="cap"``
    and the returns frame carries a ``mcap_usd``/``weight`` column.

    Returns a long frame ``[date, factor, quantile, ret]`` feeding
    :func:`long_short_stats` and the monotonicity check.
    """
    scores = as_df(scores)
    fac_cols = factor_columns(scores)

    returns = as_df(fwd_returns)
    weight_col = next((c for c in ("weight", "mcap_usd") if c in returns.columns), None)
    fwd, _ = forward_returns(returns, lags=(1,))
    fwd = fwd.rename({"_fwd1": "_fwd"})
    if weighting == "cap" and weight_col is not None:
        fwd = fwd.join(
            returns.select("stock_id", "date", weight_col), on=["stock_id", "date"]
        )

    df = scores.join(fwd, on=["stock_id", "date"], how="inner")

    if weighting == "cap" and weight_col is not None:
        w = pl.col(weight_col)
        ret_agg = ((pl.col("_fwd") * w).sum() / w.sum()).alias("ret")
    else:
        ret_agg = pl.col("_fwd").mean().alias("ret")

    frames = []
    for f in fac_cols:
        cols = [f, "_fwd"] + ([weight_col] if weighting == "cap" and weight_col else [])
        frames.append(
            df.drop_nulls(cols)
            .with_columns(_quantile_expr(f, n))
            .group_by("date", "quantile")
            .agg(ret_agg)
            .with_columns(factor=pl.lit(f))
        )

    return pl.concat(frames).select("date", "factor", "quantile", "ret").sort(
        "factor", "date", "quantile"
    )


def _series_stats(returns: np.ndarray) -> dict:
    """Cumulative return, t-stat and (per-period) Sharpe of a return series."""
    r = returns[np.isfinite(returns)]
    n = r.size
    if n < 2:
        return {"n": n, "mean": np.nan, "std": np.nan, "cum_return": np.nan,
                "t_stat": np.nan, "sharpe": np.nan}
    mean, std = float(r.mean()), float(r.std(ddof=1))
    cum = float(np.prod(1.0 + r) - 1.0)
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else np.nan
    sharpe = mean / std if std > 0 else np.nan
    return {"n": n, "mean": mean, "std": std, "cum_return": cum,
            "t_stat": t_stat, "sharpe": sharpe}


def long_short_stats(ls_returns, n=5):
    """Metrics for the Long-Short (Q1 - Qn) portfolio.

    Accepts the quantile frame from :func:`quantile_portfolios` -- from which the
    per-date long-short return is formed as quantile 1 minus quantile ``n`` (as
    specified in the Experiment Plan; flip the sign if your scores are oriented so
    high = attractive) -- or a raw series of long-short returns.

    Returns a per-factor stats frame (quantile-frame input) or a single stats dict
    (series input): number of periods, mean, std, cumulative return, t-stat and
    per-period Sharpe.
    """
    if isinstance(ls_returns, pl.DataFrame) and "quantile" in ls_returns.columns:
        keys = ["factor"] if "factor" in ls_returns.columns else []
        wide = ls_returns.pivot(values="ret", index=["date", *keys], on="quantile")
        low, high = str(1), str(n)
        wide = wide.with_columns(ls=(pl.col(low) - pl.col(high))).drop_nulls("ls")

        if not keys:
            return _series_stats(wide["ls"].to_numpy())
        rows = []
        for key_vals, sub in wide.group_by(keys, maintain_order=True):
            rows.append({"factor": key_vals[0], **_series_stats(sub["ls"].to_numpy())})
        return pl.DataFrame(rows)

    s = ls_returns if isinstance(ls_returns, pl.Series) else pl.Series(list(ls_returns))
    return _series_stats(s.to_numpy())


def fama_macbeth(scores, fwd_returns, newey_west_lags=None):
    """Aggregate cross-sectional regression premia with Newey-West t-stats.

    Runs a per-date cross-sectional OLS of the next-period return on the factor
    scores (with intercept), collects the coefficient time series, then tests each
    mean premium against zero using Newey-West (HAC) standard errors to correct
    for autocorrelation.

    Returns a frame ``[factor, mean_coef, t_stat, nw_se, n_periods]`` (``factor``
    includes ``const``). ``newey_west_lags`` defaults to the rule of thumb
    ``floor(4 (T/100)^(2/9))``.
    """
    import statsmodels.api as sm

    scores = as_df(scores)
    fac_cols = factor_columns(scores)
    fwd, _ = forward_returns(fwd_returns, lags=(1,))
    df = scores.join(fwd.rename({"_fwd1": "_y"}), on=["stock_id", "date"], how="inner")
    df = df.drop_nulls(["_y", *fac_cols])

    names = ["const", *fac_cols]
    coefs: dict[str, list[float]] = {name: [] for name in names}
    for _, sub in df.group_by("date", maintain_order=True):
        if sub.height <= len(fac_cols) + 1:  # too few obs to identify the slopes
            continue
        X = np.column_stack([np.ones(sub.height), sub.select(fac_cols).to_numpy()])
        y = sub["_y"].to_numpy().astype(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        for i, name in enumerate(names):
            coefs[name].append(float(beta[i]))

    periods = len(coefs["const"])
    if periods == 0:
        return pl.DataFrame(
            {"factor": names, "mean_coef": [np.nan] * len(names),
             "t_stat": [np.nan] * len(names), "nw_se": [np.nan] * len(names),
             "n_periods": [0] * len(names)}
        )

    if newey_west_lags is None:
        newey_west_lags = max(1, int(4 * (periods / 100) ** (2 / 9)))

    rows = []
    for name in names:
        series = np.asarray(coefs[name])
        fit = sm.OLS(series, np.ones(periods)).fit(
            cov_type="HAC", cov_kwds={"maxlags": newey_west_lags}
        )
        rows.append(
            {"factor": name, "mean_coef": float(series.mean()),
             "t_stat": float(fit.tvalues[0]), "nw_se": float(fit.bse[0]),
             "n_periods": periods}
        )
    return pl.DataFrame(rows)
