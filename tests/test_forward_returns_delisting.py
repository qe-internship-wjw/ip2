"""Terminal-return injection in forward_returns (src/validation/_common.py).

Verifies the survivorship fix: the delist-period return compounds the terminal
settlement, post-delist periods never form, and the legacy behaviour survives
behind the explicit ``delist_events=None`` opt-out.
"""

import datetime as dt

import polars as pl
import pytest

from src.validation._common import forward_returns


def _daily(sid, rows):
    return pl.DataFrame(
        {
            "stock_id": [sid] * len(rows),
            "date": [d for d, _ in rows],
            "excess_return": [r for _, r in rows],
        },
        schema={"stock_id": pl.Utf8, "date": pl.Date, "excess_return": pl.Float64},
    )


# Two quarters of prints, then a zombie tail of stale zeros after the mid-Q2 delist.
_ROWS = [
    (dt.date(2020, 1, 31), 0.0),
    (dt.date(2020, 3, 31), 0.0),
    (dt.date(2020, 4, 30), 0.10),
    (dt.date(2020, 5, 15), 0.0),
    (dt.date(2020, 6, 30), 0.0),  # zombie: after the delist date
    (dt.date(2020, 7, 31), 0.0),  # zombie: a whole phantom Q3
    (dt.date(2020, 9, 30), 0.0),
]


def _events():
    # W wipes out, Q is acquired -- both mid-Q2. GHOST is absent from the daily
    # frame; its event must be inert.
    return pl.DataFrame(
        {
            "stock_id": ["W", "Q", "GHOST"],
            "last_active_date": [dt.date(2020, 5, 15)] * 3,
            "delist_date": [dt.date(2020, 5, 15)] * 3,
            "reason": ["wipeout", "acquisition", "wipeout"],
            "delist_return": [-1.0, 0.0, -1.0],
        }
    )


def _panel():
    return pl.concat([_daily("W", _ROWS), _daily("Q", _ROWS)])


def test_delist_events_is_a_required_keyword():
    with pytest.raises(TypeError):
        forward_returns(_panel(), lags=(1,), winsorize_limits=None)


def test_terminal_return_lands_in_delist_period():
    fwd = forward_returns(
        _panel(), lags=(1, 2), winsorize_limits=None, delist_events=_events()
    ).sort("stock_id", "period")
    w = fwd.filter(pl.col("stock_id") == "W")
    q = fwd.filter(pl.col("stock_id") == "Q")

    # Post-delist zombie rows are dropped: only Q1 and Q2 exist.
    assert w["period"].to_list() == [dt.date(2020, 1, 1), dt.date(2020, 4, 1)]
    # The delist period's formation date is the last pre-delist print.
    assert w["date"].to_list() == [dt.date(2020, 3, 31), dt.date(2020, 5, 15)]

    # Wipeout: (1 + 10%) * (1 - 100%) - 1 = -1 realised one period ahead of Q1.
    assert w["_fwd1"][0] == pytest.approx(-1.0)
    # Nothing exists after the delist period.
    assert w["_fwd1"][1] is None
    assert w["_fwd2"][0] is None

    # Acquisition: terminal 0 is a no-op -- the partial-quarter return survives.
    assert q["_fwd1"][0] == pytest.approx(0.10)

    # Sign correctness for a short book: w * r flips the wipeout into a gain,
    # capped at +100% of the position.
    short_weight = -0.5
    assert short_weight * w["_fwd1"][0] == pytest.approx(0.5)


def test_opt_out_reproduces_legacy_survivorship_bias():
    fwd = forward_returns(
        _panel(), lags=(1,), winsorize_limits=None, delist_events=None
    ).sort("stock_id", "period")
    w = fwd.filter(pl.col("stock_id") == "W")

    # Zombie rows form a phantom Q3 and the terminal loss is never booked.
    assert w["period"].to_list() == [
        dt.date(2020, 1, 1),
        dt.date(2020, 4, 1),
        dt.date(2020, 7, 1),
    ]
    assert w["_fwd1"][0] == pytest.approx(0.10)
    assert w["_fwd1"][1] == pytest.approx(0.0)
