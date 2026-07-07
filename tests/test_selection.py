"""Three-step point-in-time factor selection (src/validation/selection.py).

Synthetic quarterly panels with engineered signal so each gate has a known
outcome. Column names are real registry shorthands ("NIM" is BANKS, "FIY" is
INSURANCE, "E/P" is ALL_FINANCIALS, "Beta~MKT" is the neutralize=False beta
signal, ...) so the tests exercise the registry-driven splits, not duck-typing.

Return construction: ret[:, p] responds to the factor z-scores at p-1, so the
lag-1 forward return seen at formation period s is exactly the engineered
signal of s -- strong ICs and FM premia for the true factors, noise for "DY".
"""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.validation.redundancy import select_cluster_representatives
from src.validation.selection import select_features
from src.validation.single_factor import fama_macbeth

FACTORS = ["E/P", "ln(MCap)", "DY", "TRV", "NIM", "FIY", "Beta~MKT"]

CFG = {
    "validation": {
        "ic": {"decay_lags": [1], "ir_shortlist_threshold": 0.2},
        "fama_macbeth": {"t_stat_threshold": 1.0},
        "redundancy": {
            "correlation_threshold": 0.4,
            "cluster_representative": "fm_gradient",
        },
        "selection": {"min_shortlist_warn": 0},
    },
    "backtest": {"rebalancing_frequency_months": 3},
}


def _quarters(n, start=dt.date(2000, 1, 1)):
    out, y, m = [], start.year, start.month
    for _ in range(n):
        out.append(dt.date(y, m, 1))
        m += 3
        if m > 12:
            m, y = m - 12, y + 1
    return out


def _make_data(seed=1, n_bank=30, n_ins=30, n_periods=60):
    """Panel with: E/P strong+, ln(MCap) a clone of E/P (no own premium),
    DY pure noise, TRV strong-, NIM bank-only+, FIY insurance-only+,
    Beta~MKT strong+ (neutralize=False -> raw-return IC, lasso bypass)."""
    rng = np.random.default_rng(seed)
    periods = _quarters(n_periods)
    n = n_bank + n_ins
    stocks = [f"B{i:02d}" for i in range(n_bank)] + [f"I{i:02d}" for i in range(n_ins)]
    industry = ["bank"] * n_bank + ["insurance_life"] * n_ins
    is_bank = np.array([s.startswith("B") for s in stocks])

    z = {f: rng.standard_normal((n, n_periods)) for f in FACTORS}
    z["ln(MCap)"] = 0.97 * z["E/P"] + 0.243 * rng.standard_normal((n, n_periods))

    signal = (
        0.02 * z["E/P"]
        - 0.02 * z["TRV"]
        + 0.015 * z["Beta~MKT"]
        + 0.01 * z["NIM"] * is_bank[:, None]
        + 0.01 * z["FIY"] * (~is_bank)[:, None]
    )
    ret = 0.005 * rng.standard_normal((n, n_periods))
    ret[:, 1:] += signal[:, :-1]

    # Sector factors exist only on their sub-universe (NaN -> null below).
    z["NIM"][~is_bank, :] = np.nan
    z["FIY"][is_bank, :] = np.nan

    ids = {
        "stock_id": np.repeat(stocks, n_periods),
        "date": periods * n,
        "period": periods * n,
        "industry": np.repeat(industry, n_periods),
    }
    neu = pl.DataFrame({**ids, **{f: z[f].ravel() for f in FACTORS}}).with_columns(
        [pl.col(f).fill_nan(None) for f in FACTORS]
    )
    returns = pl.DataFrame({**ids, "ret_wins": ret.ravel(), "ret_raw": ret.ravel()})
    loadings = pl.DataFrame(
        {k: ids[k] for k in ("stock_id", "date", "period")}
    ).with_columns(MKT=pl.Series(1.0 + 0.1 * rng.standard_normal(n * n_periods)))
    return neu, returns, loadings, periods


def test_select_features_end_to_end():
    neu, returns, loadings, _ = _make_data()
    res = select_features(neu, returns, loadings, CFG)

    # Final shortlist: the true factors survive all three steps; the clone and
    # the noise factor are gone; the beta signal bypasses the lasso.
    for f in ("E/P", "TRV", "NIM", "FIY", "Beta~MKT"):
        assert f in res.shortlist, f
    assert "DY" not in res.shortlist
    assert "ln(MCap)" not in res.shortlist

    # Step 1: ln(MCap) passes the gate (its IC rides on E/P's signal) -- it is
    # dropped only at clustering; DY fails the gate outright.
    sc = {row["factor"]: row for row in res.scorecard.to_dicts()}
    assert sc["ln(MCap)"]["single_pass"] and sc["ln(MCap)"]["ir_pass"]
    assert not sc["DY"]["single_pass"]

    # Step 2: the clone clusters with E/P; the larger |FM gradient| wins.
    cluster = next(c for c in res.clusters if "E/P" in c)
    assert "ln(MCap)" in cluster
    assert "E/P" in res.representatives
    assert "ln(MCap)" not in res.representatives
    assert sc["ln(MCap)"]["representative"] is False
    assert sc["ln(MCap)"]["selected"] is False
    assert sc["ln(MCap)"]["cluster_id"] == sc["E/P"]["cluster_id"]

    # Step 3: lasso ran only on the style representatives; the beta signal
    # bypasses (lasso is null) yet is selected.
    assert sc["E/P"]["lasso"] is True
    assert sc["ln(MCap)"]["lasso"] is None
    assert sc["Beta~MKT"]["lasso"] is None
    assert sc["Beta~MKT"]["selected"] is True

    # Negative premium passes the two-sided gates.
    assert sc["TRV"]["single_pass"] and sc["TRV"]["fm_t"] < 0


def test_select_features_is_point_in_time():
    neu, returns, loadings, periods = _make_data()
    cutoff = periods[39]

    res1 = select_features(neu, returns, loadings, CFG, cutoff=cutoff)
    assert len(res1.shortlist) >= 3  # meaningful selection on the window

    # Corrupt everything strictly after the cutoff: identical outputs.
    after = pl.col("period") > cutoff
    neu2 = neu.with_columns(
        [pl.when(after).then(9.9).otherwise(pl.col(f)).alias(f) for f in FACTORS]
    )
    returns2 = returns.with_columns(
        [
            pl.when(after).then(pl.col(c) * -5.0 + 1.0).otherwise(pl.col(c)).alias(c)
            for c in ("ret_wins", "ret_raw")
        ]
    )
    loadings2 = loadings.with_columns(
        pl.when(after).then(-7.0).otherwise(pl.col("MKT")).alias("MKT")
    )
    res2 = select_features(neu2, returns2, loadings2, CFG, cutoff=cutoff)

    assert res1.shortlist == res2.shortlist
    assert res1.scorecard.equals(res2.scorecard)
    assert res1.ir.equals(res2.ir)
    assert res1.fm.equals(res2.fm)


def test_select_features_train_start_and_warning():
    neu, returns, loadings, periods = _make_data()
    with pytest.warns(UserWarning, match="shortlist has"):
        res = select_features(
            neu, returns, loadings,
            {**CFG, "validation": {**CFG["validation"], "selection": {"min_shortlist_warn": 99}}},
            cutoff=periods[39], train_start=periods[8],
        )
    assert res.ic["period"].min() >= periods[8]
    assert res.ic["period"].max() <= periods[39]


def test_pooled_fama_macbeth_reports_each_factor_once():
    neu, returns, _, _ = _make_data()
    scores = neu.select("stock_id", "date", "industry", *FACTORS)

    fm = fama_macbeth(
        scores, returns, target_col="ret_wins", pooled=True, delist_events=None
    )
    fac = fm.filter(pl.col("factor") != "const")
    assert fac.group_by("factor").len()["len"].max() == 1
    sub = dict(zip(fac["factor"].to_list(), fac["sub_universe"].to_list()))
    assert sub["E/P"] == "all" and sub["Beta~MKT"] == "all"
    assert sub["NIM"] == "bank" and sub["FIY"] == "insurance"

    # Legacy architecture: all-financials factors tested once per sub-universe.
    legacy = fama_macbeth(
        scores, returns, target_col="ret_wins", pooled=False, delist_events=None
    )
    ep = legacy.filter(pl.col("factor") == "E/P")
    assert sorted(ep["sub_universe"].to_list()) == ["bank", "insurance"]


def test_pooled_fama_macbeth_pools_opposite_sign_premia():
    rng = np.random.default_rng(3)
    periods = _quarters(40)
    n_side, n = 30, 60
    stocks = [f"B{i:02d}" for i in range(n_side)] + [f"I{i:02d}" for i in range(n_side)]
    is_bank = np.array([s.startswith("B") for s in stocks])

    z = rng.standard_normal((n, 40))
    sign = np.where(is_bank, 0.03, -0.03)[:, None]
    ret = 0.002 * rng.standard_normal((n, 40))
    ret[:, 1:] += sign * z[:, :-1]

    ids = {
        "stock_id": np.repeat(stocks, 40),
        "date": periods * n,
        "period": periods * n,
        "industry": np.repeat(np.where(is_bank, "bank", "insurance_life"), 40),
    }
    scores = pl.DataFrame({**ids, "E/P": z.ravel()}).drop("period")
    returns = pl.DataFrame({**ids, "ret_wins": ret.ravel()})

    legacy = fama_macbeth(
        scores, returns, target_col="ret_wins", pooled=False, delist_events=None
    ).filter(pl.col("factor") == "E/P")
    coef = dict(zip(legacy["sub_universe"].to_list(), legacy["mean_coef"].to_list()))
    assert coef["bank"] == pytest.approx(0.03, abs=0.005)
    assert coef["insurance"] == pytest.approx(-0.03, abs=0.005)

    pooled = fama_macbeth(
        scores, returns, target_col="ret_wins", pooled=True, delist_events=None
    ).filter(pl.col("factor") == "E/P")
    assert pooled.height == 1
    assert pooled["sub_universe"][0] == "all"
    assert abs(pooled["mean_coef"][0]) < 0.01  # opposite premia cancel


def test_fm_gradient_representatives():
    fm = pl.DataFrame(
        {
            "sub_universe": ["all", "all", "bank"],
            "factor": ["A", "B", "C"],
            "mean_coef": [0.01, -0.03, float("nan")],
            "t_stat": [2.0, -2.5, float("nan")],
            "nw_se": [0.005, 0.012, float("nan")],
            "n_periods": [40, 40, 40],
        }
    )
    ir = {"A": 0.1, "B": 0.2, "C": -0.6, "D": 0.2}

    # Largest |coef| wins regardless of sign.
    assert select_cluster_representatives(
        [["A", "B"]], ic_ir=ir, fm=fm, method="fm_gradient"
    ) == ["B"]
    # No finite coefficient anywhere in the cluster -> |IR| fallback.
    assert select_cluster_representatives(
        [["C", "D"]], ic_ir=ir, fm=fm, method="fm_gradient"
    ) == ["C"]
    # A member with a coefficient always outranks one without.
    assert select_cluster_representatives(
        [["A", "C"]], ic_ir=ir, fm=fm, method="fm_gradient"
    ) == ["A"]
