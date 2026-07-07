"""Performance metrics.

Evaluate the strategy on excess return over the risk-free rate, net of trading
costs and overall market beta, plus maximum drawdown, Sharpe ratio and
information ratio.

The engine's return series is already in **excess-of-RF** space (the pipeline's
``excess_return`` target) and **quarterly** (one row per rebalance period), so:

* ``risk_free`` is normally ``None`` here -- the subtraction already happened at
  the daily level;
* ``backtest.hedge_broad_market_beta: true`` means the broad-market beta is
  assumed hedged externally at no cost. That is an **accounting adjustment
  only** (per the strategy decision, it never constrains the optimizer): pass
  the engine's per-period ex-ante ``mkt_beta`` and the quarterly-compounded
  market return to :func:`excess_return`;
* annualization defaults to ``periods_per_year=252`` (daily convention, per the
  original stub) -- **callers evaluating the engine's quarterly series must pass
  ``periods_per_year=4``**.

The scalar metrics take anything ``np.asarray`` accepts (lists, numpy, polars
Series). The **selection-dynamics metrics** at the bottom consume the dynamic
driver's outputs (:mod:`src.backtest.dynamic`): ``selection_history`` is the
stack of per-cutoff scorecards (``[cutoff, factor, ..., selected, regime]``)
and ``results`` the concatenated engine rows with a ``regime`` column.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def excess_return(returns, risk_free=None, market_beta=None, market_returns=None):
    """Return net of the risk-free rate and (optionally) the hedged market component.

    ``r_hedged_t = r_t - rf_t - beta_t * r_mkt_t``: ``beta_t`` is the portfolio's
    ex-ante market loading (``sum_i w_it * MKT_i``, reported by the engine) and
    ``r_mkt`` the market's excess return over the same periods -- the costless
    external hedge that ``backtest.hedge_broad_market_beta`` asserts. Pass
    ``market_beta`` and ``market_returns`` together or not at all.
    """
    r = np.asarray(returns, dtype=float).copy()
    if risk_free is not None:
        r = r - np.asarray(risk_free, dtype=float)
    if (market_beta is None) != (market_returns is None):
        raise ValueError("pass market_beta and market_returns together.")
    if market_beta is not None:
        r = r - np.asarray(market_beta, dtype=float) * np.asarray(
            market_returns, dtype=float
        )
    return r


def max_drawdown(returns):
    """Maximum peak-to-trough drawdown of the compounded return series (<= 0)."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return float("nan")
    curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1.0).min())


def sharpe_ratio(returns, risk_free=None, periods_per_year=252):
    """Annualized Sharpe ratio: ``mean / std * sqrt(periods_per_year)``.

    The engine's series is quarterly and already excess-of-RF: pass
    ``periods_per_year=4`` and leave ``risk_free=None``. Degenerate series
    (fewer than 2 points or zero dispersion) return NaN.
    """
    r = np.asarray(returns, dtype=float)
    if risk_free is not None:
        r = r - np.asarray(risk_free, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    std = r.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(r.mean() / std * np.sqrt(periods_per_year))


def information_ratio(returns, benchmark, periods_per_year=252):
    """Annualized information ratio of active returns vs a benchmark.

    ``mean(r - b) / std(r - b) * sqrt(periods_per_year)`` over aligned series
    (pass ``periods_per_year=4`` for the engine's quarterly output). The default
    benchmark in the tour is the cap-weighted tradeable-universe return.
    """
    active = np.asarray(returns, dtype=float) - np.asarray(benchmark, dtype=float)
    return sharpe_ratio(active, risk_free=None, periods_per_year=periods_per_year)


# ── selection-dynamics & regime metrics (dynamic backtest, DYNAMIC_SELECTION_PLAN) ──


def _frame(f) -> pl.DataFrame:
    return f.collect() if isinstance(f, pl.LazyFrame) else f


def selection_frequency(history) -> pl.DataFrame:
    """How often each factor is selected across the re-selection windows.

    Returns ``[factor, n_windows, n_selected, freq, first_selected,
    last_selected, current]`` -- ``current`` marks membership in the latest
    window's shortlist. Sorted most-selected first.
    """
    h = _frame(history)
    last_cutoff = h["cutoff"].max()
    sel = pl.col("selected")
    return (
        h.group_by("factor")
        .agg(
            n_windows=pl.len(),
            n_selected=sel.sum(),
            freq=sel.mean(),
            first_selected=pl.col("cutoff").filter(sel).min(),
            last_selected=pl.col("cutoff").filter(sel).max(),
            current=sel.filter(pl.col("cutoff") == last_cutoff).any(),
        )
        .sort(["freq", "factor"], descending=[True, False])
    )


def selection_transitions(history) -> pl.DataFrame:
    """Factor entry/exit events between consecutive selection windows.

    ``entered`` = newly selected (started working -- incl. a factor selected at
    its first-ever window); ``exited`` = previously selected, now dropped
    (stopped working). Returns ``[factor, cutoff, event]`` sorted by cutoff.
    """
    h = _frame(history).sort("factor", "cutoff")
    prev = pl.col("selected").shift(1).over("factor").fill_null(False)
    return (
        h.with_columns(
            event=pl.when(pl.col("selected") & ~prev)
            .then(pl.lit("entered"))
            .when(~pl.col("selected") & prev)
            .then(pl.lit("exited"))
            .otherwise(None)
        )
        .drop_nulls("event")
        .select("factor", "cutoff", "event")
        .sort("cutoff", "factor")
    )


def shortlist_turnover(history) -> pl.DataFrame:
    """Shortlist size and stability across consecutive selection windows.

    Returns ``[cutoff, n_selected, n_added, n_dropped, jaccard_prev]`` --
    ``jaccard_prev`` is ``|cur ∩ prev| / |cur ∪ prev|`` (1 = identical
    shortlist, 0 = full replacement); nulls on the first window.
    """
    h = _frame(history)
    cutoffs = sorted(h["cutoff"].unique().to_list())
    rows, prev = [], None
    for c in cutoffs:
        cur = set(
            h.filter((pl.col("cutoff") == c) & pl.col("selected"))["factor"].to_list()
        )
        row = {"cutoff": c, "n_selected": len(cur)}
        if prev is None:
            row.update(n_added=None, n_dropped=None, jaccard_prev=None)
        else:
            union = cur | prev
            row.update(
                n_added=len(cur - prev),
                n_dropped=len(prev - cur),
                jaccard_prev=(len(cur & prev) / len(union)) if union else None,
            )
        rows.append(row)
        prev = cur
    return pl.DataFrame(rows)


def factor_health(history) -> pl.DataFrame:
    """Long gate-statistic trajectories per factor across the windows.

    Returns ``[factor, cutoff, selected, metric, value]`` with ``metric`` in
    ``{ir_lag1, fm_t, fm_coef}`` -- the expanding-window re-tests behind the
    started/stopped-working plots.
    """
    return (
        _frame(history)
        .select("factor", "cutoff", "selected", "ir_lag1", "fm_t", "fm_coef")
        .unpivot(
            index=["factor", "cutoff", "selected"],
            variable_name="metric", value_name="value",
        )
        .sort("factor", "metric", "cutoff")
    )


def regime_performance(results, periods_per_year=4) -> pl.DataFrame:
    """Per-regime performance of the dynamic backtest's ``results`` frame.

    Returns ``[regime, start, end, n_periods, ann_ret, sharpe, max_drawdown,
    avg_turnover, avg_tc]`` (quarterly engine rows -> ``periods_per_year=4``).
    """
    df = _frame(results)
    rows = []
    for regime in sorted(df["regime"].unique().to_list()):
        sub = df.filter(pl.col("regime") == regime).sort("period")
        r = sub["net_ret"].to_numpy()
        rows.append(
            {
                "regime": int(regime),
                "start": sub["period"].min(),
                "end": sub["period"].max(),
                "n_periods": len(r),
                "ann_ret": float(np.prod(1.0 + r) ** (periods_per_year / len(r)) - 1.0),
                "sharpe": sharpe_ratio(r, periods_per_year=periods_per_year),
                "max_drawdown": max_drawdown(r),
                "avg_turnover": float(sub["turnover"].mean()),
                "avg_tc": float(sub["tc"].mean()),
            }
        )
    return pl.DataFrame(rows)
