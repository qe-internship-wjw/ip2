"""Walk-forward expected returns (src/portfolio/expected_returns.py)."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.config import Config
from src.portfolio.expected_returns import (
    expected_return_cross_section,
    integrate,
    james_stein,
    premia_series,
    strategic_weights,
    walk_forward_ic,
    walk_forward_means,
    walk_forward_premia,
)

QS = [dt.date(2020, 1, 1), dt.date(2020, 4, 1), dt.date(2020, 7, 1), dt.date(2020, 10, 1)]

CFG = Config(
    raw={
        "portfolio": {
            "strategic_allocation": "equal",
            "expected_returns": {
                "estimation": "walk_forward",
                "min_estimation_periods": 1,
                "james_stein": False,
            },
        }
    }
)


def _series(values, factor="E/P"):
    return pl.DataFrame(
        {
            "period": QS[: len(values)],
            "sub_universe": ["bank"] * len(values),
            "factor": [factor] * len(values),
            "coef": values,
        }
    )


def test_walk_forward_availability_rule():
    out = walk_forward_means(
        _series([1.0, 2.0, 3.0, 4.0]), "coef", ["sub_universe", "factor"], min_periods=1
    ).sort("period")
    # At period t only coefficients for s <= t-1 have entered.
    assert out["mean"].to_list() == [None, 1.0, 1.5, 2.0]
    assert out["n"].to_list() == [0, 1, 2, 3]

    gated = walk_forward_means(
        _series([1.0, 2.0, 3.0, 4.0]), "coef", ["sub_universe", "factor"], min_periods=2
    ).sort("period")
    assert gated["mean"].to_list() == [None, None, 1.5, 2.0]


def test_full_sample_mode_is_flat_and_leaky():
    out = walk_forward_means(
        _series([1.0, 2.0, 3.0, 4.0]), "coef", ["sub_universe", "factor"],
        mode="full_sample",
    )
    assert out["mean"].to_list() == [2.5] * 4


def test_james_stein_limits():
    def frame(means, se2):
        return pl.DataFrame(
            {
                "period": [QS[0]] * len(means),
                "factor": [f"F{i}" for i in range(len(means))],
                "mean": means,
                "se2": se2,
            }
        )

    # Zero dispersion -> full collapse (b = 1), value preserved at the grand mean.
    out = james_stein(frame([1.0] * 4, [0.01] * 4))
    assert out["js_b"].to_list() == [1.0] * 4
    assert out["mean_js"].to_list() == [1.0] * 4

    # Huge dispersion, negligible noise -> b ~ 0, estimates barely move.
    out = james_stein(frame([0.0, 10.0, -10.0, 20.0], [1e-12] * 4))
    assert out["js_b"][0] == pytest.approx(0.0, abs=1e-10)
    assert np.allclose(out["mean_js"].to_list(), [0.0, 10.0, -10.0, 20.0])

    # K <= 3 -> JS undefined -> no shrinkage.
    out = james_stein(frame([1.0, 3.0], [0.5, 0.5]))
    assert out["js_b"].to_list() == [0.0, 0.0]
    assert out["mean_js"].to_list() == [1.0, 3.0]


def test_premia_series_recovers_cross_sectional_premium():
    # Fixed z per stock; the realised next-quarter return is 0.005 + 0.02 * z
    # exactly, so every period's FM coefficient is [const=0.005, E/P=0.02].
    q_ends = [dt.date(2020, 3, 31), dt.date(2020, 6, 30),
              dt.date(2020, 9, 30), dt.date(2020, 12, 31)]
    stocks = {"B0": ("bank", -1.0), "B1": ("bank", 0.0), "B2": ("bank", 1.0),
              "I0": ("insurance_life", -1.0), "I1": ("insurance_life", 0.0),
              "I2": ("insurance_life", 1.0)}
    scores = pl.DataFrame(
        [
            {"stock_id": s, "date": d, "industry": ind, "E/P": z}
            for s, (ind, z) in stocks.items()
            for d in q_ends
        ]
    )
    daily = pl.DataFrame(
        [
            {"stock_id": s, "date": d, "excess_return": 0.005 + 0.02 * z}
            for s, (_, z) in stocks.items()
            for d in q_ends
        ]
    )
    out = premia_series(scores, daily, delist_events=None, winsorize_limits=None)
    for sub in ("bank", "insurance"):
        got = out.filter(
            (pl.col("sub_universe") == sub) & (pl.col("factor") == "E/P")
        )
        assert got.height == 3  # last quarter has no forward return
        assert np.allclose(got["coef"].to_list(), 0.02)
        const = out.filter(
            (pl.col("sub_universe") == sub) & (pl.col("factor") == "const")
        )
        assert np.allclose(const["coef"].to_list(), 0.005)


def test_walk_forward_premia_excludes_const():
    series = pl.concat([_series([1.0, 2.0, 3.0, 4.0], "E/P"),
                        _series([9.0, 9.0, 9.0, 9.0], "const")])
    out = walk_forward_premia(series, CFG)
    assert set(out["factor"].to_list()) == {"E/P"}
    latest = out.sort("period")["premium"].to_list()
    assert latest == [None, 1.0, 1.5, 2.0]


def test_walk_forward_ic_shifts_by_lag():
    ic = pl.DataFrame(
        {
            "period": QS,
            "factor": ["Momentum"] * 4,
            "lag": [2] * 4,
            "ic": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = walk_forward_ic(ic, CFG, lag=2).sort("period")
    # A lag-2 IC for period s is only known at s+2.
    assert out["ic"].to_list() == [None, None, 1.0, 1.5]


def test_expected_return_cross_section_combines_kinds():
    neu = pl.DataFrame(
        {
            "stock_id": ["B1", "I1"],
            "industry": ["bank", "insurance_life"],
            "E/P": [1.0, -1.0],       # systematic (registry)
            "Momentum": [2.0, None],  # behavioural (registry); null -> no bet
        }
    )
    premia = pl.DataFrame(
        {
            "sub_universe": ["bank", "insurance"],
            "factor": ["E/P", "E/P"],
            "premium": [0.02, 0.01],
        }
    )
    ic = pl.DataFrame({"factor": ["Momentum"], "ic": [0.05]})
    sigma = pl.DataFrame({"stock_id": ["B1", "I1"], "sigma": [0.3, 0.2]})

    out = expected_return_cross_section(neu, premia, ic, sigma, CFG)
    mu = dict(zip(out["stock_id"].to_list(), out["mu"].to_list()))
    # Equal alpha = 1/2 over {E/P, Momentum}.
    assert mu["B1"] == pytest.approx(0.5 * 0.02 * 1.0 + 0.5 * 0.05 * 0.3 * 2.0)
    assert mu["I1"] == pytest.approx(0.5 * 0.01 * -1.0)


def test_strategic_weights_and_integrate():
    assert strategic_weights(["A", "B"], CFG) == {"A": 0.5, "B": 0.5}

    ir_cfg = Config(raw={"portfolio": {"strategic_allocation": "ir_weighted"}})
    alpha = strategic_weights(["A", "B"], ir_cfg, ir={"A": 0.3, "B": -0.1})
    assert alpha["A"] == pytest.approx(0.75)
    assert alpha["B"] == pytest.approx(-0.25)
    with pytest.raises(ValueError):
        strategic_weights(["A"], ir_cfg)  # ir_weighted without IRs

    mu = integrate(
        {"A": [0.01, np.nan], "B": [0.0, 0.02]}, {"A": 0.5, "B": 0.5}
    )
    assert np.allclose(mu, [0.005, 0.01])
