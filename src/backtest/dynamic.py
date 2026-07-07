"""Dynamic-selection walk-forward driver (DYNAMIC_SELECTION_PLAN.md §1.2).

Per re-selection regime (:func:`~src.backtest.schedule.regime_schedule`):

1. **Select** -- :func:`~src.validation.selection.select_features` at the
   regime's cutoff (data sliced to ``period <= cutoff``, no look-ahead).
2. **Re-estimate beliefs** -- :func:`~src.backtest.inputs.build_inputs` rebuilds
   the FM premia / IC series and the monthly risk histories on the regime's
   factor set over the full history (each per-period observation is PIT; the
   engine slices ``<= t``).
3. **Run the segment** -- :func:`~src.backtest.engine.run` bounded to the
   regime's formation periods, with the previous segment's
   :class:`~src.backtest.engine.EngineState` threaded in: the first rebalance
   trades against the carried drifted book (turnover charged on the change) and
   the risk EMA aligns by factor name across the shortlist switch.

An empty selection carries the previous regime's shortlist forward (flagged in
the diagnostics) rather than trading an empty book; a shortlist below
``validation.selection.min_shortlist_warn`` is flagged, never enforced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ..validation.selection import select_features
from .engine import run as engine_run
from .inputs import Artifacts, build_inputs
from .metrics import excess_return
from .schedule import Regime, regime_schedule


@dataclass
class DynamicResult:
    """Concatenated segment outputs + the per-regime selection record."""

    results: pl.DataFrame            # engine rows + `regime`
    weights: pl.DataFrame            # [period, stock_id, weight, regime]
    selection_history: pl.DataFrame  # stacked scorecards + `regime`
    shortlists: dict                 # cutoff ISO string -> traded factor list
    regimes: list[Regime] = field(default_factory=list)
    diagnostics: list = field(default_factory=list)


def run_dynamic(cfg, artifacts: Artifacts, solver_opts=None) -> DynamicResult:
    """Select -> rebuild inputs -> run, per regime, with threaded engine state."""
    regimes = regime_schedule(cfg)
    sched = cfg.get("backtest", {}).get("schedule") or {}
    train_start = sched.get("train_start")
    min_warn = int(
        (cfg.get("validation", {}).get("selection") or {}).get("min_shortlist_warn", 0)
    )

    state = None
    prev_shortlist: list | None = None
    results, weights, history, diags = [], [], [], []
    shortlists: dict = {}

    for reg in regimes:
        sel = select_features(
            artifacts.neu_q, artifacts.returns_q, artifacts.loadings_q, cfg,
            cutoff=reg.cutoff, train_start=train_start,
        )
        shortlist, carried = sel.shortlist, False
        if not shortlist:
            if prev_shortlist is None:
                raise RuntimeError(
                    f"regime {reg.index} ({reg.cutoff}): empty shortlist and no "
                    "previous regime to carry forward."
                )
            shortlist, carried = prev_shortlist, True
        shortlists[reg.cutoff.isoformat()] = shortlist
        history.append(
            sel.scorecard.with_columns(regime=pl.lit(reg.index, dtype=pl.Int64))
        )
        diags.append(
            {
                "regime": reg.index,
                "cutoff": str(reg.cutoff),
                "n_selected": len(sel.shortlist),
                "carried_forward": carried,
                "below_min_warn": bool(min_warn and len(shortlist) < min_warn),
                "shortlist": list(shortlist),
            }
        )

        inputs = build_inputs(shortlist, cfg, artifacts)
        res = engine_run(
            inputs, cfg, solver_opts=solver_opts,
            start_period=reg.formation_start, end_period=reg.formation_end,
            initial_state=state,
        )
        state = res.state
        if res.results.height:
            results.append(
                res.results.with_columns(regime=pl.lit(reg.index, dtype=pl.Int64))
            )
            weights.append(
                res.weights.with_columns(regime=pl.lit(reg.index, dtype=pl.Int64))
            )
        diags.extend({"regime": reg.index, **d} for d in res.diagnostics)
        prev_shortlist = shortlist

    if not results:
        raise RuntimeError("run_dynamic: no regime produced a traded rebalance.")
    return DynamicResult(
        results=pl.concat(results),
        weights=pl.concat(weights),
        selection_history=pl.concat(history),
        shortlists=shortlists,
        regimes=regimes,
        diagnostics=diags,
    )


def performance_frame(results, market_daily, cfg) -> pl.DataFrame:
    """Results + quarterly-compounded market/benchmark + the hedged net series.

    Each engine row's P&L realizes over the *following* period, so the market
    and benchmark returns are joined on ``pnl_period = period + 1``. ``hedged``
    is the accounting-only broad-market hedge
    (:func:`~src.backtest.metrics.excess_return` with the engine's ex-ante
    ``mkt_beta``); when ``backtest.hedge_broad_market_beta`` is off it equals
    ``net_ret``. The tour notebook renders this frame as-is.
    """
    pm = int(cfg.get("backtest", {}).get("rebalancing_frequency_months", 3))
    market_daily = (
        market_daily.collect()
        if isinstance(market_daily, pl.LazyFrame)
        else market_daily
    )
    q = (
        market_daily.with_columns(period=pl.col("date").dt.truncate(f"{pm}mo"))
        .group_by("period")
        .agg(
            mkt=(pl.col("mkt") + 1.0).product() - 1.0,
            bench=(pl.col("bench") + 1.0).product() - 1.0,
        )
    )
    perf = (
        results.with_columns(pnl_period=pl.col("period").dt.offset_by(f"{pm}mo"))
        .join(q, left_on="pnl_period", right_on="period", how="left")
        .sort("pnl_period")
    )
    if cfg.get("backtest", {}).get("hedge_broad_market_beta", True):
        hedged = excess_return(
            perf["net_ret"].to_numpy(),
            market_beta=perf["mkt_beta"].fill_null(0.0).to_numpy(),
            market_returns=perf["mkt"].fill_null(0.0).to_numpy(),
        )
    else:
        hedged = perf["net_ret"].to_numpy()
    return perf.with_columns(hedged=pl.Series(hedged))
