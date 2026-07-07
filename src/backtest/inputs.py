"""Assemble :class:`~src.backtest.engine.BacktestInputs` from the processed artifacts.

This is the input-plumbing that used to live in ``backtest_tour.ipynb`` cells
5-9, rebased onto the ``data/processed`` feathers (``scripts/build_processed.py``)
so no panel is ever rebuilt: the quarterly cross-sections come from
``neu_{pm}m`` / ``loadings_{pm}m``, the monthly risk histories from the ``_1m``
variants, returns from the periodic panels (``ret_wins`` for estimation,
``ret_raw`` for realized P&L -- trim/winsorization/delisting settlement baked
in), and free float from the returns panel's period-end meta columns.

``build_inputs`` is shortlist-parameterized: the dynamic driver calls it once
per regime with that regime's selected factors, recomputing the FM premia
series (pooled architecture, DYNAMIC_SELECTION_PLAN.md §1.5), the lag-1 IC
series (beta signals against raw returns) and the monthly factor-return /
residual histories on the active set -- "re-estimating beliefs" at each
re-selection. Every per-period observation uses only contemporaneous data, so
the engine's slice-``<= t`` discipline keeps each regime point-in-time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..factors.base import registry
from ..portfolio.expected_returns import premia_series
from ..portfolio.risk_model import factor_returns
from ..portfolio.transaction_cost import free_float_mcap
from ..validation._common import forward_returns, impute_loadings
from ..validation.single_factor import rank_ic
from .engine import BacktestInputs

_ID_COLS = ("stock_id", "date", "period")


@dataclass
class Artifacts:
    """The ``data/processed`` frames the backtest consumes (quarterly + monthly)."""

    neu_q: pl.DataFrame        # neu_{pm}m: [stock_id, date, period, industry, <z cols>]
    loadings_q: pl.DataFrame   # loadings_{pm}m: [stock_id, date, period, MKT, beta_*, is_*]
    returns_q: pl.DataFrame    # returns_{pm}m: [.., industry, mcap_usd, free_float_percentage, ret_raw, ret_wins]
    neu_m: pl.DataFrame        # monthly variants (risk-model histories)
    loadings_m: pl.DataFrame
    returns_m: pl.DataFrame
    delist_events: pl.DataFrame
    market_daily: pl.DataFrame  # [date, mkt, bench] (metrics / hedging)


def load_artifacts(cfg, project_root=None) -> Artifacts:
    """Read every artifact the backtest needs from ``data.cache``."""
    cache = Path(cfg["data"].get("cache", "data/processed"))
    if project_root is not None and not cache.is_absolute():
        cache = Path(project_root) / cache
    pm = int(cfg["backtest"]["rebalancing_frequency_months"])
    fm = int(cfg["portfolio"]["risk_model"]["frequency_months"])

    def read(name: str) -> pl.DataFrame:
        path = cache / f"{name}.feather"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing -- run scripts/build_processed.py first."
            )
        return pl.read_ipc(path)

    return Artifacts(
        neu_q=read(f"neu_{pm}m"), loadings_q=read(f"loadings_{pm}m"),
        returns_q=read(f"returns_{pm}m"),
        neu_m=read(f"neu_{fm}m"), loadings_m=read(f"loadings_{fm}m"),
        returns_m=read(f"returns_{fm}m"),
        delist_events=read("delist_events"), market_daily=read("market_daily"),
    )


def risk_exposures(neu, loadings, shortlist, design_cols) -> pl.DataFrame:
    """The risk model's exposure block: shortlist z + structural design.

    ``neu ⋈ loadings`` on ``(stock_id, period)``; rolling-beta warm-up nulls in
    the design are median-imputed per period, then every exposure fills null
    with 0 (a z is defined as neutral off its sub-universe).
    """
    expo_cols = [*shortlist, *design_cols]
    e = neu.select("stock_id", "period", "industry", *shortlist).join(
        loadings.drop("date"), on=["stock_id", "period"], how="inner"
    )
    e = impute_loadings(e, list(design_cols), by="period")
    return e.with_columns([pl.col(c).fill_null(0.0) for c in expo_cols]).select(
        "stock_id", "period", "industry", *expo_cols
    )


def build_inputs(shortlist, cfg, artifacts: Artifacts) -> BacktestInputs:
    """Everything :func:`src.backtest.engine.run` needs, for one factor set.

    ``delist_events=None`` in the ``forward_returns`` / ``premia_series`` /
    ``rank_ic`` calls below is *inert*: the periodic artifacts already carry the
    delisting trim and terminal settlement. The engine still receives the events
    frame itself (PIT investable-set gate + exit-cost classification).
    """
    shortlist = list(shortlist)
    reg = registry()
    unknown = [f for f in shortlist if f not in reg]
    if unknown:
        raise ValueError(f"build_inputs: unregistered factors {unknown}")
    design = [c for c in artifacts.loadings_q.columns if c not in _ID_COLS]
    expo_cols = [*shortlist, *design]

    expo_q = risk_exposures(artifacts.neu_q, artifacts.loadings_q, shortlist, design)
    expo_m = risk_exposures(artifacts.neu_m, artifacts.loadings_m, shortlist, design)

    # Monthly risk regressions: month-m exposures on the (delisting-adjusted,
    # winsorized) return realized over m+1.
    fwd_m = forward_returns(
        artifacts.returns_m, lags=(1,), target_col="ret_wins", delist_events=None
    )
    risk_frame = expo_m.join(
        fwd_m.select("stock_id", "period", pl.col("_fwd1").alias("_y")),
        on=["stock_id", "period"], how="inner",
    )
    f_rets, resids = factor_returns(risk_frame, expo_cols, "_y")

    # Per-period FM coefficient series on the active set, pooled architecture
    # (period-keyed join: the returns panel's delist-trimmed dates need not
    # match the score grid's).
    premia = premia_series(
        artifacts.neu_q.select("stock_id", "period", "industry", *shortlist),
        artifacts.returns_q, target_col="ret_wins", delist_events=None, pooled=True,
    )

    # Lag-1 IC series: style factors against loadings-residualised returns,
    # beta signals (neutralize=False) against raw returns (stage-5 caveat).
    style = [f for f in shortlist if reg[f].neutralize]
    beta = [f for f in shortlist if not reg[f].neutralize]
    ic_frames = []
    if style:
        ic_frames.append(rank_ic(
            artifacts.neu_q.select("stock_id", "date", *style), artifacts.returns_q,
            lags=(1,), nonstyle_exposures=artifacts.loadings_q,
            target_col="ret_wins", delist_events=None,
        ))
    if beta:
        ic_frames.append(rank_ic(
            artifacts.neu_q.select("stock_id", "date", *beta), artifacts.returns_q,
            lags=(1,), nonstyle_exposures=None,
            target_col="ret_wins", delist_events=None,
        ))
    ic = pl.concat(ic_frames) if ic_frames else None

    realized = forward_returns(
        artifacts.returns_q, lags=(1,), target_col="ret_raw", delist_events=None
    )
    ff_q = free_float_mcap(
        artifacts.returns_q.select(
            "stock_id", "date", "period", "industry",
            "mcap_usd", "free_float_percentage",
        ),
        by=["period", "industry"],
    ).select("stock_id", "period", "free_float_mcap")

    return BacktestInputs(
        neu=artifacts.neu_q.select("stock_id", "period", "industry", *shortlist),
        exposures=expo_q, realized=realized, premia=premia,
        factor_rets=f_rets, residuals=resids, free_float=ff_q,
        ic=ic, delist_events=artifacts.delist_events, shortlist=shortlist,
    )
