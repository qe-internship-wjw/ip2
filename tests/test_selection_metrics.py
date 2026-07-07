"""Selection-dynamics & regime metrics (src/backtest/metrics.py extension)."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.backtest.metrics import (
    factor_health,
    regime_performance,
    selection_frequency,
    selection_transitions,
    shortlist_turnover,
)

C1, C2, C3 = dt.date(2005, 10, 1), dt.date(2007, 10, 1), dt.date(2009, 10, 1)

# A: always selected; B: dropped after the first window; C: enters at the second.
HISTORY = pl.DataFrame(
    {
        "cutoff": [C1] * 3 + [C2] * 3 + [C3] * 3,
        "factor": ["A", "B", "C"] * 3,
        "selected": [True, True, False, True, False, True, True, False, True],
        "ir_lag1": [0.5, 0.3, 0.1, 0.4, 0.1, 0.3, 0.45, 0.05, 0.35],
        "fm_t": [3.0, 1.5, 0.2, 2.5, 0.4, 1.8, 2.8, 0.1, 2.0],
        "fm_coef": [0.02, 0.01, 0.001, 0.018, 0.002, 0.012, 0.02, 0.0, 0.014],
    }
)


def test_selection_frequency():
    freq = {r["factor"]: r for r in selection_frequency(HISTORY).to_dicts()}
    assert freq["A"]["n_selected"] == 3 and freq["A"]["freq"] == 1.0
    assert freq["A"]["current"] is True
    assert freq["B"]["n_selected"] == 1
    assert freq["B"]["first_selected"] == C1 and freq["B"]["last_selected"] == C1
    assert freq["B"]["current"] is False
    assert freq["C"]["first_selected"] == C2 and freq["C"]["current"] is True
    # Sorted most-selected first.
    assert selection_frequency(HISTORY)["factor"][0] == "A"


def test_selection_transitions():
    events = {
        (r["factor"], r["cutoff"]): r["event"]
        for r in selection_transitions(HISTORY).to_dicts()
    }
    assert events == {
        ("A", C1): "entered",
        ("B", C1): "entered",
        ("B", C2): "exited",
        ("C", C2): "entered",
    }


def test_shortlist_turnover():
    rows = {r["cutoff"]: r for r in shortlist_turnover(HISTORY).to_dicts()}
    assert rows[C1]["n_selected"] == 2 and rows[C1]["jaccard_prev"] is None
    # {A, B} -> {A, C}: one added, one dropped, |{A}| / |{A,B,C}|.
    assert rows[C2]["n_added"] == 1 and rows[C2]["n_dropped"] == 1
    assert rows[C2]["jaccard_prev"] == pytest.approx(1 / 3)
    # {A, C} -> {A, C}: unchanged.
    assert rows[C3]["n_added"] == 0 and rows[C3]["jaccard_prev"] == 1.0


def test_factor_health_long_format():
    health = factor_health(HISTORY)
    assert health.height == 9 * 3
    a_t = health.filter(
        (pl.col("factor") == "A") & (pl.col("metric") == "fm_t")
    ).sort("cutoff")
    assert a_t["value"].to_list() == [3.0, 2.5, 2.8]


def test_regime_performance():
    q = [dt.date(2006, 1 + 3 * i, 1) for i in range(4)]
    results = pl.DataFrame(
        {
            "regime": [0, 0, 1, 1],
            "period": q,
            "net_ret": [0.10, -0.05, 0.02, 0.02],
            "turnover": [0.5, 0.1, 0.4, 0.1],
            "tc": [0.001, 0.0002, 0.0008, 0.0002],
        }
    )
    perf = {r["regime"]: r for r in regime_performance(results).to_dicts()}
    assert perf[0]["n_periods"] == 2 and perf[0]["start"] == q[0]
    assert perf[0]["ann_ret"] == pytest.approx((1.10 * 0.95) ** 2 - 1)
    assert perf[0]["max_drawdown"] == pytest.approx(-0.05)
    assert perf[1]["max_drawdown"] == 0.0
    assert perf[1]["sharpe"] != perf[1]["sharpe"] or perf[1]["sharpe"] > 0  # NaN-safe
    assert perf[0]["avg_turnover"] == pytest.approx(0.3)
    np.testing.assert_allclose(perf[1]["avg_tc"], 0.0005)