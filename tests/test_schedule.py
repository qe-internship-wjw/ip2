"""Regime schedule for the dynamic-selection backtest (src/backtest/schedule.py)."""

import datetime as dt

import pytest

from src.backtest.schedule import Regime, regime_schedule


def _cfg(**overrides):
    sched = {
        "train_start": dt.date(2000, 1, 1),
        "trade_start": dt.date(2006, 1, 1),
        "end": dt.date(2025, 12, 31),
        "reselection_frequency_months": 24,
        **overrides,
    }
    return {"backtest": {"rebalancing_frequency_months": 3, "schedule": sched}}


def test_default_schedule_ten_regimes():
    regimes = regime_schedule(_cfg())
    assert len(regimes) == 10
    assert regimes[0] == Regime(
        index=0, cutoff=dt.date(2005, 10, 1),
        formation_start=dt.date(2005, 10, 1), formation_end=dt.date(2007, 7, 1),
    )
    # First formation books the P&L of the first trade quarter (2006Q1); the
    # last books 2025Q4.
    assert regimes[-1].cutoff == dt.date(2023, 10, 1)
    assert regimes[-1].formation_end == dt.date(2025, 7, 1)
    # Contiguous, non-overlapping 8-quarter regimes.
    for prev, nxt in zip(regimes, regimes[1:]):
        assert nxt.cutoff == dt.date(
            prev.cutoff.year + 2, prev.cutoff.month, prev.cutoff.day
        )
        assert prev.formation_end < nxt.formation_start


def test_partial_final_regime():
    regimes = regime_schedule(_cfg(end=dt.date(2024, 6, 30)))
    assert regimes[-1].cutoff == dt.date(2023, 10, 1)
    # end period 2024Q2 -> last formation 2024Q1: a 2-quarter partial regime.
    assert regimes[-1].formation_end == dt.date(2024, 1, 1)


def test_iso_strings_accepted():
    cfg = _cfg(trade_start="2006-01-01", end="2025-12-31")
    assert regime_schedule(cfg)[0].cutoff == dt.date(2005, 10, 1)


def test_missing_schedule_raises():
    with pytest.raises(KeyError, match="backtest.schedule"):
        regime_schedule({"backtest": {"rebalancing_frequency_months": 3}})


def test_end_before_start_raises():
    with pytest.raises(ValueError, match="precedes"):
        regime_schedule(_cfg(end=dt.date(2005, 1, 1)))
