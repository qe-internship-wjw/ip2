"""Engine segmentation, name-aligned risk-EMA state, and the pooled mu path.

The invariance test is the correctness gate for the dynamic driver: a
monolithic engine run and the equivalent segmented runs with threaded
EngineState must be bit-identical (same books, same P&L) -- regime boundaries
are then pure bookkeeping, never an economic event.
"""

import datetime as dt

import numpy as np
import polars as pl
import pytest
from test_engine import CFG, OPTS, QS, _inputs

from src.backtest.engine import run
from src.portfolio.expected_returns import (
    expected_return_cross_section,
    premia_series,
)
from src.portfolio.risk_model import estimate

RCFG = {
    "portfolio": {
        "risk_model": {
            "frequency_months": 1,
            "factor_cov_window_months": 24,
            "idio_window_months": 24,
            "idio_min_obs": 6,
        },
        "covariance": {"ema_coefficient": 0.9},
    },
    "backtest": {"rebalancing_frequency_months": 3},
}


# ── engine segmentation ───────────────────────────────────────────────────────


def test_segmented_run_matches_monolithic():
    full = run(_inputs(), CFG, solver_opts=OPTS)

    cut = QS[4]
    first = run(_inputs(), CFG, solver_opts=OPTS, end_period=cut)
    second = run(
        _inputs(), CFG, solver_opts=OPTS,
        start_period=QS[5], initial_state=first.state,
    )

    # The split point sits inside the traded window on both sides.
    assert first.results["period"].max() == cut
    assert second.results["period"].min() == QS[5]

    assert pl.concat([first.results, second.results]).equals(full.results)
    assert pl.concat([first.weights, second.weights]).equals(full.weights)


def test_segment_without_state_differs_from_monolithic():
    # Dropping the threaded state rebuilds the book from cash: the first
    # rebalance of the second segment must differ (turnover, hence net_ret).
    full = run(_inputs(), CFG, solver_opts=OPTS)
    cold = run(_inputs(), CFG, solver_opts=OPTS, start_period=QS[5])

    t = QS[5]
    row_full = full.results.filter(pl.col("period") == t).to_dicts()[0]
    row_cold = cold.results.filter(pl.col("period") == t).to_dicts()[0]
    assert row_cold["turnover"] > row_full["turnover"]


def test_final_state_reflects_drifted_book():
    res = run(_inputs(), CFG, solver_opts=OPTS)
    last = res.weights.filter(pl.col("period") == QS[-1])
    assert res.state is not None and res.state.risk_state is not None
    # Every surviving position of the last book appears in the carried state.
    held = {sid for sid, w in last.select("stock_id", "weight").iter_rows() if abs(w) > 1e-9}
    assert held <= set(res.state.prev_book)


# ── risk-EMA state aligned by factor name ─────────────────────────────────────


def _risk_history(factors, seed, n_stocks=6, n_months=30):
    rng = np.random.default_rng(seed)
    months = [dt.date(2020 + m // 12, m % 12 + 1, 1) for m in range(n_months)]
    stocks = [f"S{i}" for i in range(n_stocks)]
    exposures = pl.DataFrame(
        {"stock_id": stocks, **{f: rng.standard_normal(n_stocks) for f in factors}}
    )
    rets = pl.DataFrame(
        [
            {"period": m, "factor": f, "f": float(rng.standard_normal() * 0.02)}
            for m in months
            for f in [*factors, "const"]
        ]
    )
    resid = pl.DataFrame(
        [
            {"stock_id": s, "period": m, "resid": float(rng.standard_normal() * 0.05)}
            for m in months
            for s in stocks
        ]
    )
    return exposures, rets, resid


def test_risk_ema_aligns_by_factor_name_on_a_swap():
    exp_a, rets_a, resid_a = _risk_history(["x1", "x2"], seed=0)
    rm_a = estimate(exp_a, rets_a, resid_a, RCFG)

    # Same K, one factor swapped (x2 -> x3): shape alone cannot tell them apart.
    exp_b, rets_b, resid_b = _risk_history(["x1", "x3"], seed=1)
    rm_fresh = estimate(exp_b, rets_b, resid_b, RCFG)
    rm_b = estimate(exp_b, rets_b, resid_b, RCFG, state=rm_a.state)

    ia = rm_a.factors.index("x1")
    ib, jb = rm_b.factors.index("x1"), rm_b.factors.index("x3")
    # The survivor keeps its EMA history...
    assert rm_b.F[ib, ib] == pytest.approx(0.9 * rm_a.F[ia, ia] + 0.1 * rm_fresh.F[ib, ib])
    # ...the newcomer seeds at the fresh windowed estimate, x2's history untouched.
    assert rm_b.F[jb, jb] == pytest.approx(rm_fresh.F[jb, jb])
    # Cross terms involving the newcomer are fresh too.
    assert rm_b.F[ib, jb] == pytest.approx(rm_fresh.F[ib, jb])

    # An unchanged factor set stays a plain EMA (fixed point on identical inputs).
    rm_same = estimate(exp_a, rets_a, resid_a, RCFG, state=rm_a.state)
    assert np.allclose(rm_same.F, rm_a.F)


def test_legacy_unlabelled_state_still_accepted():
    exp_a, rets_a, resid_a = _risk_history(["x1", "x2"], seed=0)
    rm_a = estimate(exp_a, rets_a, resid_a, RCFG)
    legacy = (rm_a.F, pl.DataFrame({"stock_id": rm_a.stock_ids, "_d": rm_a.D}))
    rm_same = estimate(exp_a, rets_a, resid_a, RCFG, state=legacy)
    assert np.allclose(rm_same.F, rm_a.F)
    assert np.allclose(rm_same.D, rm_a.D)


# ── pooled premia in the mu path ──────────────────────────────────────────────


def test_expected_return_applies_pooled_premium_to_every_row():
    neu_t = pl.DataFrame(
        {
            "stock_id": ["B0", "I0"],
            "industry": ["bank", "insurance_life"],
            "E/P": [1.0, -1.0],
        }
    )
    premia_t = pl.DataFrame(
        {"sub_universe": ["all"], "factor": ["E/P"], "premium": [0.02]}
    )
    mu = dict(
        expected_return_cross_section(neu_t, premia_t, None, None, {}).iter_rows()
    )
    assert mu["B0"] == pytest.approx(0.02)
    assert mu["I0"] == pytest.approx(-0.02)


def test_expected_return_sub_premium_overrides_pooled():
    neu_t = pl.DataFrame(
        {
            "stock_id": ["B0", "I0"],
            "industry": ["bank", "insurance_life"],
            "E/P": [1.0, 1.0],
        }
    )
    premia_t = pl.DataFrame(
        {
            "sub_universe": ["all", "bank"],
            "factor": ["E/P", "E/P"],
            "premium": [0.02, 0.05],
        }
    )
    mu = dict(
        expected_return_cross_section(neu_t, premia_t, None, None, {}).iter_rows()
    )
    assert mu["B0"] == pytest.approx(0.05)  # own-sub premium wins
    assert mu["I0"] == pytest.approx(0.02)  # falls back to the pooled premium


def test_premia_series_pooled_reports_each_factor_once():
    rng = np.random.default_rng(5)
    periods = [dt.date(2020, 3 * q + 1, 1) for q in range(4)]
    stocks = [f"B{i}" for i in range(10)] + [f"I{i}" for i in range(10)]
    is_bank = np.array([s.startswith("B") for s in stocks])
    n, p = len(stocks), len(periods)

    ids = {
        "stock_id": np.repeat(stocks, p),
        "period": periods * n,
        "industry": np.repeat(np.where(is_bank, "bank", "insurance_life"), p),
    }
    z_ep = rng.standard_normal((n, p))
    z_nim = rng.standard_normal((n, p))
    z_nim[~is_bank, :] = np.nan
    scores = pl.DataFrame(
        {**ids, "E/P": z_ep.ravel(), "NIM": z_nim.ravel()}
    ).with_columns(pl.col("NIM").fill_nan(None))
    returns = pl.DataFrame(
        {
            **ids,
            "date": periods * n,
            "ret_wins": rng.standard_normal(n * p) * 0.01,
        }
    )

    pooled = premia_series(
        scores, returns, delist_events=None, target_col="ret_wins", pooled=True
    )
    fac = pooled.filter(pl.col("factor") != "const")
    subs = {
        f: sorted(fac.filter(pl.col("factor") == f)["sub_universe"].unique().to_list())
        for f in ("E/P", "NIM")
    }
    assert subs["E/P"] == ["all"]
    assert subs["NIM"] == ["bank"]

    legacy = premia_series(
        scores, returns, delist_events=None, target_col="ret_wins", pooled=False
    )
    assert sorted(
        legacy.filter(pl.col("factor") == "E/P")["sub_universe"].unique().to_list()
    ) == ["bank", "insurance"]