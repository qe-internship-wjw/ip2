"""Walk-forward backtest engine (src/backtest/engine.py).

A controlled synthetic world: one all-financials factor ("E/P"), fixed z per
stock, constant per-period FM premium 0.02 and hand-set realized returns, so
every engine number is recomputable by hand.
"""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.backtest.engine import BacktestInputs, run
from src.config import Config

QS = [dt.date(2020, 1, 1), dt.date(2020, 4, 1), dt.date(2020, 7, 1),
      dt.date(2020, 10, 1), dt.date(2021, 1, 1), dt.date(2021, 4, 1),
      dt.date(2021, 7, 1), dt.date(2021, 10, 1)]
MONTHS = [dt.date(2019 + m // 12, m % 12 + 1, 1) for m in range(36)]  # 2019-01..2021-12

# stock -> (industry, z, per-period realized forward return)
BASE = {
    "B0": ("bank", 1.0, 0.04),
    "B1": ("bank", -1.0, -0.02),
    "I0": ("insurance_life", 1.0, 0.03),
    "I1": ("insurance_life", -1.0, -0.01),
}

OPTS = {"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iter": 200_000}

CFG = Config(
    raw={
        "portfolio": {
            "strategic_allocation": "equal",
            "risk_aversion": 2.0,
            "expected_returns": {
                "estimation": "walk_forward",
                "min_estimation_periods": 2,
                "james_stein": False,
            },
            "constraints": {
                "gross_leverage": 2.0,
                "net_exposure": [-0.2, 0.2],
                "max_name_weight": None,
            },
            "risk_model": {
                "frequency_months": 1,
                "factor_cov_window_months": 24,
                "idio_window_months": 24,
                "idio_min_obs": 6,
            },
            "covariance": {"ema_coefficient": 0.9},
            "optimizer": {"solver": "OSQP"},
        },
        "backtest": {"rebalancing_frequency_months": 3},
    }
)


def _q_end(p):
    m = p.month - 1 + 3
    nxt = dt.date(p.year + m // 12, m % 12 + 1, 1)
    return nxt - dt.timedelta(days=1)


def _inputs(extra=(), events=None):
    """``extra``: tuples (sid, industry, z, periods, fwd_by_period)."""
    stocks = [(sid, ind, z, QS, {p: f for p in QS}) for sid, (ind, z, f) in BASE.items()]
    stocks += list(extra)

    neu_rows, realized_rows, ff_rows = [], [], []
    for sid, ind, z, periods, fwd in stocks:
        for p in periods:
            neu_rows.append({"stock_id": sid, "period": p, "industry": ind, "E/P": z})
            ff_rows.append({"stock_id": sid, "period": p, "free_float_mcap": 1e10})
            if fwd.get(p) is not None:
                realized_rows.append(
                    {"stock_id": sid, "date": _q_end(p), "period": p, "_fwd1": fwd[p]}
                )
    neu = pl.DataFrame(neu_rows)

    premia = pl.DataFrame(
        [
            {"period": p, "sub_universe": sub, "factor": f, "coef": c}
            for p in QS
            for sub in ("bank", "insurance")
            for f, c in (("E/P", 0.02), ("const", 0.0))
        ]
    )
    rng = np.random.RandomState(7)
    factor_rets = pl.DataFrame(
        [
            {"period": m, "factor": f, "f": float(rng.randn() * 0.02)}
            for m in MONTHS
            for f in ("E/P", "const")
        ]
    )
    residuals = pl.DataFrame(
        [
            {"stock_id": sid, "period": m, "resid": float(rng.randn() * 0.05)}
            for m in MONTHS
            for sid, *_ in stocks
        ]
    )
    return BacktestInputs(
        neu=neu,
        exposures=neu,  # the style z doubles as the risk exposure in this world
        realized=pl.DataFrame(realized_rows),
        premia=premia,
        factor_rets=factor_rets,
        residuals=residuals,
        free_float=pl.DataFrame(ff_rows),
        delist_events=events,
    )


def _returns_lookup(inputs):
    return {
        (r["stock_id"], r["period"]): r["_fwd1"]
        for r in inputs.realized.to_dicts()
    }


def _check_consistency(res, inputs):
    """gross_ret must equal sum(w * r) of the recorded book; net = gross - tc."""
    r_map = _returns_lookup(inputs)
    for row in res.results.to_dicts():
        w_t = res.weights.filter(pl.col("period") == row["period"])
        gross = sum(
            w * r_map.get((sid, row["period"]), 0.0)
            for sid, w in zip(w_t["stock_id"].to_list(), w_t["weight"].to_list())
        )
        assert row["gross_ret"] == pytest.approx(gross, abs=1e-10)
        assert row["net_ret"] == pytest.approx(row["gross_ret"] - row["tc"], abs=1e-12)


def test_warmup_constraints_signs_and_consistency():
    inputs = _inputs()
    res = run(inputs, CFG, solver_opts=OPTS)

    # Warm-up: premia need 2 observed quarters -> first trade at q3.
    traded = res.results["period"].to_list()
    assert traded == QS[2:]
    skipped = {d["period"]: d["skipped"] for d in res.diagnostics if "skipped" in d}
    assert set(skipped) == set(QS[:2])

    # Constraints honoured every period (loose OSQP feasibility tolerance).
    assert (res.results["gross_lev"] <= 2.0 + 1e-2).all()
    assert res.results["net_exp"].abs().max() <= 0.2 + 1e-2

    # mu = 0.02 * z: long the +1 names, short the -1 names.
    w3 = dict(
        res.weights.filter(pl.col("period") == QS[2])
        .select("stock_id", "weight")
        .iter_rows()
    )
    assert w3["B0"] > 0 and w3["I0"] > 0
    assert w3["B1"] < 0 and w3["I1"] < 0

    _check_consistency(res, inputs)

    # Same mu each quarter and slow-moving risk: the L1 cost holds the book
    # nearly still after the initial build.
    tv = res.results["turnover"].to_list()
    assert max(tv[1:]) < 0.5 * tv[0]

    # All books earn: +z names have positive returns, -z names negative.
    assert res.results["net_ret"].min() > 0


def test_delisting_exit_uncharged_but_voluntary_exit_charged():
    # W: acquisition-delisted mid-q5 (terminal 0 booked in its q4 return);
    # rows still exist at q5 to prove the PIT gate, not the data, excludes it.
    w_fwd = {p: 0.02 for p in QS[:4]}  # q5+: no realized rows
    w = ("W", "bank", 1.0, QS[:5], w_fwd)
    events = pl.DataFrame(
        {
            "stock_id": ["W"],
            "last_active_date": [dt.date(2021, 2, 15)],
            "delist_date": [dt.date(2021, 2, 15)],
            "reason": ["acquisition"],
            "delist_return": [0.0],
        }
    )
    # V: still listed but leaves the data after q4 -> voluntary sale at q5.
    v = ("V", "insurance_life", 1.0, QS[:4], {p: 0.02 for p in QS[:4]})

    res_w = run(_inputs(extra=[w], events=events), CFG, solver_opts=OPTS)
    res_v = run(_inputs(extra=[v]), CFG, solver_opts=OPTS)

    q4, q5 = QS[3], QS[4]
    for res, sid in ((res_w, "W"), (res_v, "V")):
        held = res.weights.filter(
            (pl.col("period") == q4) & (pl.col("stock_id") == sid)
        )["weight"]
        assert held[0] > 0  # positively scored, so held at q4
        assert res.weights.filter(
            (pl.col("period") == q5) & (pl.col("stock_id") == sid)
        ).height == 0  # gone at q5

    tc_exit_w = dict(res_w.results.select("period", "tc_exit").iter_rows())
    tc_exit_v = dict(res_v.results.select("period", "tc_exit").iter_rows())
    assert tc_exit_w[q5] == 0.0  # involuntary: settles without a trade
    assert tc_exit_v[q5] > 0.0  # voluntary: a real sale pays its coefficient


def test_wipeout_dents_long_book_and_pays_short_book():
    # Both wipe out mid-q5; the q4 realized return carries the terminal -100%.
    events = pl.DataFrame(
        {
            "stock_id": ["WL", "WS"],
            "last_active_date": [dt.date(2021, 2, 15)] * 2,
            "delist_date": [dt.date(2021, 2, 15)] * 2,
            "reason": ["wipeout"] * 2,
            "delist_return": [-1.0] * 2,
        }
    )
    wl = ("WL", "bank", 1.0, QS[:5], {**{p: 0.02 for p in QS[:3]}, QS[3]: -1.0})
    ws = ("WS", "insurance_life", -1.0, QS[:5], {**{p: -0.02 for p in QS[:3]}, QS[3]: -1.0})
    inputs = _inputs(extra=[wl, ws], events=events)
    res = run(inputs, CFG, solver_opts=OPTS)

    q4 = QS[3]
    w4 = dict(
        res.weights.filter(pl.col("period") == q4)
        .select("stock_id", "weight")
        .iter_rows()
    )
    assert w4["WL"] > 0 and w4["WS"] < 0
    # Sign-aware settlement: the long eats -100%, the short books the gain.
    assert w4["WL"] * -1.0 < 0 < w4["WS"] * -1.0
    _check_consistency(res, inputs)

    # Involuntary exits: nothing charged at q5; the wiped-out long drifted to 0.
    tc_exit = dict(res.results.select("period", "tc_exit").iter_rows())
    assert tc_exit[QS[4]] == 0.0


def test_nav_wipeout_resets_the_book():
    # M1 (shorted, z = -1) gaps +500% in q3: a ~1/3 short loses ~1.7x NAV.
    # The dead book must restart from cash, never renormalize by ~zero into
    # astronomically leveraged garbage weights (the 1e17-gross failure mode).
    fwd = {p: -0.01 for p in QS}
    fwd[QS[2]] = 5.0
    moon = ("M1", "insurance_life", -1.0, QS, fwd)
    res = run(_inputs(extra=[moon]), CFG, solver_opts=OPTS)

    q3 = QS[2]
    row = res.results.filter(pl.col("period") == q3).to_dicts()[0]
    assert row["net_ret"] < -1.0  # the loss itself stays on the books
    assert any(d.get("period") == q3 and "book_wiped" in d for d in res.diagnostics)
    # Every subsequent book is rebuilt from cash with sane leverage.
    after = res.results.filter(pl.col("period") > q3)
    assert after.height > 0
    assert (after["gross_lev"] <= 2.0 + 1e-2).all()
    assert np.isfinite(res.weights["weight"].to_numpy()).all()


def test_no_leakage_from_future_inputs():
    base = run(_inputs(), CFG, solver_opts=OPTS)

    # Corrupt everything from q6 / 2021-06 onward, far beyond what any
    # rebalance <= q5 may read (risk bound at q5 is months < 2021-03; premia
    # at q5 average coefficients for s <= q4).
    tampered = _inputs()
    q6, m6 = QS[5], dt.date(2021, 6, 1)
    tampered.premia = tampered.premia.with_columns(
        coef=pl.when(pl.col("period") >= q6).then(pl.col("coef") * 100).otherwise(pl.col("coef"))
    )
    tampered.factor_rets = tampered.factor_rets.with_columns(
        f=pl.when(pl.col("period") >= m6).then(pl.col("f") * 100).otherwise(pl.col("f"))
    )
    tampered.residuals = tampered.residuals.with_columns(
        resid=pl.when(pl.col("period") >= m6).then(pl.col("resid") * 100).otherwise(pl.col("resid"))
    )
    tampered.realized = tampered.realized.with_columns(
        _fwd1=pl.when(pl.col("period") >= q6).then(pl.col("_fwd1") + 1.0).otherwise(pl.col("_fwd1"))
    )
    tam = run(tampered, CFG, solver_opts=OPTS)

    for t in QS[2:5]:  # every traded period before the corruption
        a = base.weights.filter(pl.col("period") == t).sort("stock_id")
        b = tam.weights.filter(pl.col("period") == t).sort("stock_id")
        assert a["stock_id"].to_list() == b["stock_id"].to_list()
        assert np.allclose(a["weight"].to_numpy(), b["weight"].to_numpy())
