"""periodic_returns / forward_from_periodic split and the periodic dispatch
in forward_returns (src/validation/_common.py).

The daily path (trim -> winsorize -> compound -> settle -> shift) must be
exactly reproducible from a persisted periodic panel: that is the contract
scripts/build_processed.py relies on.
"""

import datetime as dt

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from src.validation._common import (
    forward_from_periodic,
    forward_returns,
    periodic_returns,
)


def _daily():
    """Two stocks, 2020 H1, two prints per month; B delists mid-May."""
    dates = [dt.date(2020, m, d) for m in range(1, 7) for d in (10, 20)]
    rows = []
    for sid, r in (("A", 0.01), ("B", 0.02)):
        for i, d in enumerate(dates):
            rows.append(
                {
                    "stock_id": sid,
                    "date": d,
                    "excess_return": r,
                    "mcap_usd": 100.0 + i if sid == "A" else 200.0 + i,
                    "industry": "bank" if sid == "A" else "insurance_life",
                }
            )
    return pl.DataFrame(rows)


def _events():
    return pl.DataFrame(
        {
            "stock_id": ["B"],
            "delist_date": [dt.date(2020, 5, 15)],
            "delist_return": [-1.0],
        }
    )


def test_forward_returns_is_periodic_plus_shift():
    daily, events = _daily(), _events()
    direct = forward_returns(
        daily, lags=(1, 2), weight_col="mcap_usd", delist_events=events
    )
    composed = forward_from_periodic(
        periodic_returns(daily, weight_col="mcap_usd", delist_events=events),
        lags=(1, 2),
    )
    assert_frame_equal(direct, composed)


def test_periodic_settlement_and_trim():
    per = periodic_returns(_daily(), delist_events=_events())
    b = per.filter(pl.col("stock_id") == "B").sort("period")
    # Zombie tail dropped: no period forms after the delist quarter.
    assert b["period"].to_list() == [dt.date(2020, 1, 1), dt.date(2020, 4, 1)]
    # Wipeout terminal booked in the delist quarter.
    assert b["_ret"][-1] == pytest.approx(-1.0)
    # Delisting-aware period end: last pre-delist print, not the calendar quarter end.
    assert b["date"][-1] == dt.date(2020, 5, 10)
    # The survivor compounds untouched.
    a = per.filter(pl.col("stock_id") == "A").sort("period")
    assert a["_ret"][0] == pytest.approx(1.01**6 - 1.0)


def test_periodic_dispatch_matches_daily_path():
    daily, events = _daily(), _events()
    base = forward_returns(daily, lags=(1,), delist_events=events)
    per = periodic_returns(daily, delist_events=events).rename({"_ret": "ret_wins"})
    # Precomputed panel: construction kwargs are inert, target_col picks the column.
    disp = forward_returns(
        per, lags=(1,), target_col="ret_wins", delist_events=None
    )
    assert_frame_equal(base, disp)


def test_periodic_dispatch_requires_target_col():
    per = periodic_returns(_daily(), delist_events=None).rename({"_ret": "ret_wins"})
    with pytest.raises(ValueError, match="ret_wins"):
        forward_returns(per, target_col="excess_return", delist_events=None)


def test_weight_col_accepts_sequence():
    per = periodic_returns(
        _daily(), weight_col=("mcap_usd", "industry"), delist_events=None
    )
    assert {"mcap_usd", "industry"} <= set(per.columns)
    a_q1 = per.filter(
        (pl.col("stock_id") == "A") & (pl.col("period") == dt.date(2020, 1, 1))
    )
    # Levels sampled at the period end (.last()), never compounded.
    assert a_q1["mcap_usd"][0] == 105.0  # 100 + index of 2020-03-20
    assert a_q1["industry"][0] == "bank"
    # And the sequence passes through the forward shift too.
    fwd = forward_returns(
        _daily(), lags=(1,), weight_col=("mcap_usd", "industry"), delist_events=None
    )
    assert {"mcap_usd", "industry", "_fwd1"} <= set(fwd.columns)
