"""Backtest engine: the walk-forward rebalance loop.

The engine **slices, it never estimates**: every input is a precomputed artifact
whose observations only use contemporaneous data (the precompute-then-slice
pattern, PORTFOLIO_PLAN.md §0), so point-in-time discipline reduces to slicing
``<= t`` inside the loop. Per quarterly rebalance ``t``:

1. **Risk model** -- LW + EMA :func:`~src.portfolio.risk_model.estimate` on the
   monthly histories available at formation (months ``m`` whose return month
   ``m+1`` has completed by the end of quarter ``t``), state threaded through.
2. **Expected returns** -- the walk-forward premia/IC tables (availability rule
   ``s <= t-1``) sliced at ``t``; ``sigma_i`` from step 1; integrate -> ``mu_t``.
3. **Investable set** -- names in the ``t`` cross-section with >= 1 non-null z and
   a risk exposure row, PIT-gated: a name whose ``delist_date`` falls in period
   ``t`` or earlier is excluded (never trade a dead name), while delisted names
   remain in *history* -- survivorship bias lives in dropping them retroactively.
4. **Optimize** -- MVO with L1 costs against the drifted previous weights.
5. **Account** -- gross return ``sum_i w_it * _fwd1_it`` on the delisting-adjusted,
   unwinsorized realized returns (a mid-holding wipeout books its terminal -100%
   here, sign-aware for shorts); TC = optimizer turnover cost + **voluntary** exit
   cost for names that left the investable set while still listed. Involuntary
   (delisting) exits are never charged; their freed notional sits in cash until
   the next rebalance.
6. **Drift** -- ``w * (1 + r) / (1 + r_net)`` becomes the next ``w_prev`` (a wiped
   out name drifts to exactly 0 via its terminal return).

Row ``t`` of the output holds the book formed at ``t`` and the P&L realized over
period ``t+1``. Warm-up is implicit: periods where no premium/IC is available yet
(``min_estimation_periods``) or the risk model lacks residual history are skipped.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..portfolio.expected_returns import (
    expected_return_cross_section,
    strategic_weights,
    walk_forward_ic,
    walk_forward_means,
    walk_forward_premia,
)
from ..portfolio.optimizer import solve
from ..portfolio.risk_model import estimate
from ..portfolio.transaction_cost import cost
from ..validation._common import as_df, factor_columns


@dataclass
class BacktestInputs:
    """Precomputed pipeline artifacts the engine slices per rebalance.

    All ``period`` keys are calendar-bucket starts (``date.dt.truncate``):
    quarterly for the rebalance frames, monthly for the risk histories.
    """

    neu: pl.DataFrame            # [stock_id, period, industry, <style z cols>] quarterly
    exposures: pl.DataFrame      # [stock_id, period, (industry,), <risk exposure cols>] quarterly
    realized: pl.DataFrame       # [stock_id, date, period, _fwd1] delisting-adjusted, unwinsorized
    premia: pl.DataFrame         # [period, sub_universe, factor, coef] per-period FM series
    factor_rets: pl.DataFrame    # [period(monthly), factor, f] risk factor returns
    residuals: pl.DataFrame      # [stock_id, period(monthly), resid] risk residuals
    free_float: pl.DataFrame     # [stock_id, period, free_float_mcap] quarterly
    ic: pl.DataFrame | None = None            # [period, factor, lag, ic] rank-IC series
    delist_events: pl.DataFrame | None = None  # [stock_id, ..., delist_date, delist_return]
    shortlist: list | None = None              # style factors for mu (default: registry cols of neu)


@dataclass
class EngineState:
    """Book + risk-EMA state carried across segmented runs (regime boundaries).

    ``prev_book`` is the drifted post-P&L book of the last traded rebalance
    (``stock_id -> weight``); ``risk_state`` is the factor-labelled
    :class:`~src.portfolio.risk_model.RiskState`. Thread a segment's final
    state into the next segment's ``initial_state`` so the first rebalance
    there trades *against* the carried book (turnover charged on the change)
    instead of rebuilding from cash, and the risk EMA stays continuous.
    """

    prev_book: dict = field(default_factory=dict)
    risk_state: object | None = None


@dataclass
class BacktestResult:
    results: pl.DataFrame        # one row per traded rebalance (see `run`)
    weights: pl.DataFrame        # [period, stock_id, weight]
    diagnostics: list = field(default_factory=list)
    state: EngineState | None = None  # final state (segmented-run threading)


def _add_months(d: dt.date, k: int) -> dt.date:
    m = d.month - 1 + k
    return dt.date(d.year + m // 12, m % 12 + 1, 1)


def _utf8_ids(frame):
    """Solver outputs carry plain-string ids; normalize slices so joins align."""
    return frame.with_columns(pl.col("stock_id").cast(pl.Utf8))


def _ir_series(ic_series, cfg):
    """Walk-forward IR (mean/std of the IC series) per (period, factor)."""
    ecfg = cfg.get("portfolio", {}).get("expected_returns", {}) or {}
    wf = walk_forward_means(
        as_df(ic_series).filter(pl.col("lag") == 1), "ic", ["factor"],
        min_periods=int(ecfg.get("min_estimation_periods", 8)),
        shift_periods=1, mode=ecfg.get("estimation", "walk_forward"),
    )
    return wf.with_columns(
        ir=pl.col("mean") / (pl.col("se2") * pl.col("n")).sqrt()
    ).select("period", "factor", "ir")


def run(
    inputs: BacktestInputs,
    cfg,
    solver_opts=None,
    *,
    start_period=None,
    end_period=None,
    initial_state: EngineState | None = None,
) -> BacktestResult:
    """Run the walk-forward backtest; see the module docstring for the loop.

    ``start_period`` / ``end_period`` (inclusive period keys) bound the
    rebalance iteration so a long run can be split at regime boundaries;
    ``initial_state`` threads the previous segment's drifted book and risk-EMA
    state in (see :class:`EngineState`). A monolithic run and the equivalent
    segmented runs with threaded state produce identical results.

    Returns a :class:`BacktestResult` whose ``results`` frame has one row per
    traded rebalance: ``[period, date, gross_ret, tc_trade, tc_exit, tc,
    net_ret, turnover, gross_lev, net_exp, n_long, n_short, n_names,
    n_return_gaps, mkt_beta]`` -- ``gross_ret``/``net_ret`` are realized over
    the *following* period; ``mkt_beta`` is the ex-ante portfolio market
    loading (null when the exposures carry no ``MKT`` column) for the
    accounting-only hedge in :mod:`.metrics`. ``result.state`` holds the final
    :class:`EngineState` for the next segment.
    """
    pm = int(cfg.get("backtest", {}).get("rebalancing_frequency_months", 3))
    idio_min_obs = int(
        cfg.get("portfolio", {}).get("risk_model", {}).get("idio_min_obs", 12)
    )
    allocation = cfg.get("portfolio", {}).get("strategic_allocation", "equal")

    neu = _utf8_ids(as_df(inputs.neu))
    exposures = _utf8_ids(as_df(inputs.exposures))
    realized = _utf8_ids(as_df(inputs.realized))
    residuals = _utf8_ids(as_df(inputs.residuals))
    free_float = _utf8_ids(as_df(inputs.free_float))
    factor_rets = as_df(inputs.factor_rets)

    style_cols = inputs.shortlist or factor_columns(neu)
    if not style_cols:
        raise ValueError("run: no registered style factor columns in `neu`.")
    neu = neu.select("stock_id", "period", "industry", *style_cols)

    # Walk-forward estimate tables, computed once and sliced per rebalance.
    lams = walk_forward_premia(inputs.premia, cfg)
    ics = walk_forward_ic(inputs.ic, cfg, lag=1) if inputs.ic is not None else None
    irs = None
    if allocation == "ir_weighted":
        if inputs.ic is None:
            raise ValueError("strategic_allocation=ir_weighted needs the IC series.")
        irs = _ir_series(inputs.ic, cfg)

    delist_periods = None
    if inputs.delist_events is not None:
        delist_periods = _utf8_ids(as_df(inputs.delist_events)).select(
            "stock_id",
            _delist_period=pl.col("delist_date").dt.truncate(f"{pm}mo"),
        )

    periods = sorted(
        set(neu["period"].to_list()) & set(realized["period"].to_list())
    )
    if start_period is not None:
        periods = [t for t in periods if t >= start_period]
    if end_period is not None:
        periods = [t for t in periods if t <= end_period]

    prev_book: dict[str, float] = (
        dict(initial_state.prev_book) if initial_state is not None else {}
    )
    state = initial_state.risk_state if initial_state is not None else None
    rows, weight_rows, diags = [], [], []

    for t in periods:
        # ── monthly risk histories available at formation: an observation for
        # month m regresses on the return over m+1, known once m+1 has completed
        # -- i.e. m + 1mo < start of quarter t+1  <=>  m < t + (pm - 1) months.
        bound = _add_months(t, pm - 1)
        fr_t = factor_rets.filter(pl.col("period") < bound)
        res_t = residuals.filter(pl.col("period") < bound)
        if (
            fr_t["period"].n_unique() < 2
            or res_t["period"].n_unique() < idio_min_obs
        ):
            diags.append({"period": t, "skipped": "risk warm-up"})
            continue

        # ── walk-forward mu inputs at t (availability rule already applied).
        premia_t = lams.filter(pl.col("period") == t).drop_nulls("premium")
        ic_t = (
            ics.filter(pl.col("period") == t).drop_nulls("ic")
            if ics is not None
            else None
        )
        if premia_t.height == 0 and (ic_t is None or ic_t.height == 0):
            diags.append({"period": t, "skipped": "premia/IC warm-up"})
            continue

        # ── realized holding returns of positions formed at t; the last
        # (right-censored) formation has nothing to realize -- do not trade it.
        rt = realized.filter(pl.col("period") == t).select("stock_id", "date", "_fwd1")
        if rt.drop_nulls("_fwd1").height == 0:
            diags.append({"period": t, "skipped": "no realized forward returns"})
            continue

        # ── investable set: >=1 non-null z, an exposure row, and PIT-alive.
        cross = neu.filter(pl.col("period") == t)
        if delist_periods is not None:
            cross = cross.join(
                delist_periods.filter(pl.col("_delist_period") <= t),
                on="stock_id", how="anti",
            )
        cross = cross.filter(
            pl.any_horizontal([pl.col(c).is_not_null() for c in style_cols])
        )
        expos = exposures.filter(pl.col("period") == t).join(
            cross.select("stock_id"), on="stock_id", how="semi"
        )
        if expos.height == 0:
            diags.append({"period": t, "skipped": "empty cross-section"})
            continue
        cross = cross.join(expos.select("stock_id"), on="stock_id", how="semi")

        # ── 1. risk model (EMA state threads across rebalances).
        rm = estimate(expos.drop("period"), fr_t, res_t, cfg, state=state)
        state = rm.state

        # ── 2. expected returns; sigma_i comes from the risk model just built.
        sigma = expos.select("stock_id").with_columns(
            pl.Series("sigma", rm.sigma_idio)
        )
        alpha = None
        if irs is not None:
            ir_t = {
                r["factor"]: r["ir"]
                for r in irs.filter(pl.col("period") == t).drop_nulls("ir").to_dicts()
            }
            alpha = strategic_weights(style_cols, cfg, ir=ir_t)
        mu_df = expected_return_cross_section(cross, premia_t, ic_t, sigma, cfg, alpha=alpha)
        mu_map = dict(zip(mu_df["stock_id"].to_list(), mu_df["mu"].to_list()))
        if len(mu_map) != len(rm.stock_ids):
            raise RuntimeError(f"mu/risk misalignment at {t}: {len(mu_map)} vs {len(rm.stock_ids)}")
        mu = np.array([mu_map[sid] for sid in rm.stock_ids])

        ff_map = dict(
            free_float.filter(pl.col("period") == t)
            .select("stock_id", "free_float_mcap")
            .iter_rows()
        )
        mff = np.array([ff_map.get(sid, np.nan) for sid in rm.stock_ids])

        # ── exits: names held before but not investable now. Involuntary
        # (delisted) exits settle without a trade -- never charged; voluntary
        # leavers are real sales at their cost coefficient.
        investable = set(rm.stock_ids)
        dead = (
            set(
                delist_periods.filter(pl.col("_delist_period") <= t)["stock_id"].to_list()
            )
            if delist_periods is not None
            else set()
        )
        tc_exit = 0.0
        for sid in [s for s in prev_book if s not in investable]:
            w_exit = prev_book.pop(sid)
            if sid not in dead:
                tc_exit += cost([w_exit], [ff_map.get(sid, np.nan)])

        # ── 3. optimize against the drifted previous book.
        w_prev = np.array([prev_book.get(sid, 0.0) for sid in rm.stock_ids])
        weights, diag = solve(mu, rm, w_prev, mff, cfg, solver_opts=solver_opts)
        w = weights["weight"].to_numpy()
        tc_trade = diag["tc"]
        tc = tc_trade + tc_exit

        # ── 4. accrue over the following period; null forward return on a
        # still-listed name is a data gap: exit at formation price (0), counted.
        acc = weights.join(rt, on="stock_id", how="left")
        n_gaps = acc.filter(
            pl.col("_fwd1").is_null() & (pl.col("weight").abs() > 1e-12)
        ).height
        r = acc["_fwd1"].fill_null(0.0).to_numpy()
        gross_ret = float(w @ r)
        net_ret = gross_ret - tc

        # ── 5. drift into the next w_prev (NAV renormalization, net of costs).
        denom = 1.0 + net_ret
        if denom <= 1e-3:
            # NAV wiped out (a shorted name can gap up by far more than 100%):
            # renormalizing by ~0 would explode the book into garbage weights.
            # The dead book restarts from cash at the next rebalance; the loss
            # itself is already on the books via net_ret.
            prev_book = {}
            diags.append({"period": t, "book_wiped": net_ret})
        else:
            drifted = w * (1.0 + r) / denom
            prev_book = {
                sid: float(wd)
                for sid, wd in zip(weights["stock_id"].to_list(), drifted)
                if abs(wd) > 1e-12
            }

        mkt_beta = (
            float(w @ expos["MKT"].fill_null(0.0).to_numpy())
            if "MKT" in expos.columns
            else None
        )
        rows.append(
            {
                "period": t,
                "date": rt["date"].max(),
                "gross_ret": gross_ret,
                "tc_trade": tc_trade,
                "tc_exit": tc_exit,
                "tc": tc,
                "net_ret": net_ret,
                "turnover": diag["turnover"],
                "gross_lev": diag["gross"],
                "net_exp": diag["net"],
                "n_long": int((w > 1e-12).sum()),
                "n_short": int((w < -1e-12).sum()),
                "n_names": len(rm.stock_ids),
                "n_return_gaps": n_gaps,
                "mkt_beta": mkt_beta,
            }
        )
        weight_rows.extend(
            {"period": t, "stock_id": sid, "weight": float(wi)}
            for sid, wi in zip(weights["stock_id"].to_list(), w)
        )
        diags.append({"period": t, **{k: diag[k] for k in ("status", "solver", "solved")}, **rm.meta})

    return BacktestResult(
        results=pl.DataFrame(rows),
        weights=pl.DataFrame(weight_rows),
        diagnostics=diags,
        state=EngineState(prev_book=dict(prev_book), risk_state=state),
    )
