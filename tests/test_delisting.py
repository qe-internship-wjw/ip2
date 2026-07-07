"""Delisting classifier & PIT tradeability window (src/data/delisting.py).

Synthetic price paths per DELISTING_HANDLING.md part 2: wipeout, acquisition-like,
zombie/stale tail, floor sentinel, and right-censored (alive) records.
"""

import datetime as dt

import polars as pl

from src.config import Config
from src.data.delisting import delist_events, pit_filter, tradeability_window

START = dt.date(2020, 1, 1)
N_DAYS = 400  # panel end = 2021-02-03

CFG = Config(
    raw={
        "backtest": {
            "delisting_return": -1.0,
            "delisting": {
                "drawdown_threshold": 0.30,
                "peak_window_days": 252,
                "floor_price": 1e-5,
                "censor_buffer_days": 15,
            },
        }
    }
)


def _frame(sid, prices, volumes, returns):
    dates = [START + dt.timedelta(days=i) for i in range(len(prices))]
    return pl.DataFrame(
        {
            "stock_id": [sid] * len(prices),
            "date": dates,
            "price_local": prices,
            "price_return": returns,
            "volume": volumes,
        },
        schema={
            "stock_id": pl.Utf8,
            "date": pl.Date,
            "price_local": pl.Float64,
            "price_return": pl.Float64,
            "volume": pl.Float64,
        },
    )


def _auto_returns(prices):
    return [None] + [p1 / p0 - 1.0 for p0, p1 in zip(prices, prices[1:])]


def _panel():
    frames = []

    # Alive to the panel end -> right-censored, no event.
    prices = [100.0] * N_DAYS
    frames.append(_frame("ALIVE", prices, [100.0] * N_DAYS, _auto_returns(prices)))

    # Stops 8 days before the panel end -> inside the censor buffer, no event.
    prices = [100.0] * (N_DAYS - 8)
    frames.append(_frame("LATE", prices, [10.0] * len(prices), _auto_returns(prices)))

    # Wipeout: 100 flat days, then a linear bleed 100 -> 1 (99% drawdown), silent stop.
    prices = [100.0] * 100 + [100.0 - 99.0 * (j + 1) / 100 for j in range(100)]
    frames.append(_frame("WIPE", prices, [10.0] * 200, _auto_returns(prices)))

    # Acquisition-like: flat at 100, record simply stops mid-panel.
    prices = [100.0] * 250
    frames.append(_frame("ACQ", prices, [10.0] * 250, _auto_returns(prices)))

    # Zombie tail: real trading (bleeding 100 -> 5) for 150 days, then 250 rows of
    # flat carry-forward with zero volume/returns. Exit must be judged at day 149.
    live = [100.0 - 95.0 * (j + 1) / 150 for j in range(150)]
    prices = live + [5.0] * 250
    returns = _auto_returns(live) + [0.0] * 250
    volumes = [10.0] * 150 + [0.0] * 250
    frames.append(_frame("ZOMB", prices, volumes, returns))

    # Floor sentinel: flat at 100 while trading, then a null-return tail at 1e-6.
    # Drawdown test sees exit 100 vs peak 100 (no fire); the floor test must fire.
    prices = [100.0] * 180 + [1e-6] * 220
    returns = [None] * 180 + [None] * 220
    volumes = [10.0] * 180 + [0.0] * 220
    frames.append(_frame("FLOR", prices, volumes, returns))

    return pl.concat(frames)


def test_classifier_reasons_and_censoring():
    ev = delist_events(_panel().lazy(), CFG)
    rows = {r["stock_id"]: r for r in ev.to_dicts()}

    # Censored names emit no event.
    assert set(rows) == {"WIPE", "ACQ", "ZOMB", "FLOR"}

    assert rows["WIPE"]["reason"] == "wipeout"
    assert rows["WIPE"]["delist_return"] == -1.0
    assert rows["WIPE"]["delist_date"] == START + dt.timedelta(days=199)

    assert rows["ACQ"]["reason"] == "acquisition"
    assert rows["ACQ"]["delist_return"] == 0.0

    # Zombie tail trimmed: the exit is the last *active* day, not the last row.
    assert rows["ZOMB"]["delist_date"] == START + dt.timedelta(days=149)
    assert rows["ZOMB"]["reason"] == "wipeout"

    # Floor sentinel fires even though the drawdown test does not (and the sentinel
    # sits in the post-active tail).
    assert rows["FLOR"]["reason"] == "wipeout"
    assert rows["FLOR"]["delist_date"] == START + dt.timedelta(days=179)


def test_classifier_knobs_are_read_from_config():
    # Absurdly tight thresholds turn the 99%-drawdown exit into an "acquisition".
    cfg = Config(
        raw={
            "backtest": {
                "delisting_return": -1.0,
                "delisting": {
                    "drawdown_threshold": 0.005,
                    "peak_window_days": 252,
                    "floor_price": 1e-9,
                    "censor_buffer_days": 15,
                },
            }
        }
    )
    ev = delist_events(_panel().lazy(), cfg)
    rows = {r["stock_id"]: r for r in ev.to_dicts()}
    assert rows["WIPE"]["reason"] == "acquisition"
    assert rows["WIPE"]["delist_return"] == 0.0


def test_defaults_used_when_config_section_missing():
    ev = delist_events(_panel().lazy(), Config(raw={}))
    rows = {r["stock_id"]: r for r in ev.to_dicts()}
    assert rows["WIPE"]["reason"] == "wipeout"
    assert rows["WIPE"]["delist_return"] == -1.0


def test_tradeability_window_and_pit_filter():
    events = pl.DataFrame(
        {
            "stock_id": ["D"],
            "last_active_date": [dt.date(2020, 6, 30)],
            "delist_date": [dt.date(2020, 6, 30)],
            "reason": ["wipeout"],
            "delist_return": [-1.0],
        }
    )
    master = pl.DataFrame(
        {
            "stock_id": ["D", "A"],
            "stock_start_date": [dt.date(2019, 1, 1), dt.date(2020, 3, 1)],
        }
    )
    window = tradeability_window(master.lazy(), events)

    panel = pl.DataFrame(
        {
            "stock_id": ["D", "D", "D", "A", "A", "X"],
            "date": [
                dt.date(2018, 12, 1),  # D before listing -> drop
                dt.date(2020, 1, 15),  # D live -> keep
                dt.date(2020, 7, 15),  # D after delist -> drop
                dt.date(2020, 2, 1),   # A before listing -> drop
                dt.date(2020, 4, 1),   # A live (no delist) -> keep
                dt.date(2020, 1, 1),   # X unknown to master -> keep
            ],
        }
    )
    out = pit_filter(panel, window)
    assert isinstance(out, pl.DataFrame)  # eager in -> eager out
    kept = set(zip(out["stock_id"].to_list(), out["date"].to_list()))
    assert kept == {
        ("D", dt.date(2020, 1, 15)),
        ("A", dt.date(2020, 4, 1)),
        ("X", dt.date(2020, 1, 1)),
    }

    lazy_out = pit_filter(panel.lazy(), window)
    assert isinstance(lazy_out, pl.LazyFrame)  # lazy in -> lazy out
    assert lazy_out.collect().height == out.height
