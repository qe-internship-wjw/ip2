"""Regime schedule for the dynamic-selection walk-forward backtest.

Derives from ``backtest.schedule`` the selection cutoffs and formation-period
ranges of each re-selection regime (DYNAMIC_SELECTION_PLAN.md §1.3). All dates
are rebalance-period **starts** (the ``period`` key,
``date.dt.truncate("{pm}mo")``).

Formation convention (matches the engine): the book formed at period ``t``
realizes its P&L over ``t+1``. So the first formation period sits one rebalance
period *before* ``trade_start`` (the first P&L period), and the last formation
period books the P&L of the ``end`` period. Selection at a regime's ``cutoff``
(its first formation period) uses data through the end of that period --
:func:`src.validation.selection.select_features` slices ``period <= cutoff``
before the forward shift, so only returns realised by the formation date enter.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


def _add_months(d: dt.date, months: int) -> dt.date:
    y, m = divmod(d.year * 12 + d.month - 1 + months, 12)
    return dt.date(y, m + 1, d.day)


def _truncate(d: dt.date, period_months: int) -> dt.date:
    """The start of the ``period_months`` calendar bucket containing ``d``."""
    m0 = ((d.month - 1) // period_months) * period_months
    return dt.date(d.year, m0 + 1, 1)


def _as_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    raise TypeError(f"expected a date, datetime or ISO string, got {type(value)!r}")


@dataclass(frozen=True)
class Regime:
    """One re-selection regime: ``cutoff`` doubles as the first formation period."""

    index: int
    cutoff: dt.date
    formation_start: dt.date
    formation_end: dt.date  # inclusive


def regime_schedule(cfg) -> list[Regime]:
    """The re-selection regimes implied by ``backtest.schedule``.

    With the default config (trade 2006-2025 quarterly, re-select every 24
    months) this yields 10 regimes of 8 formation quarters each, cutoffs
    2005Q4, 2007Q4, ..., 2023Q4. A final regime shorter than the re-selection
    interval is kept (partial regimes trade until ``end``).
    """
    bt = cfg.get("backtest", {}) or {}
    sched = bt.get("schedule") or {}
    if not sched:
        raise KeyError(
            "regime_schedule needs the backtest.schedule config block "
            "(train_start / trade_start / end / reselection_frequency_months)."
        )
    pm = int(bt.get("rebalancing_frequency_months", 3))
    resel = int(sched.get("reselection_frequency_months", 24))

    first_formation = _add_months(_truncate(_as_date(sched["trade_start"]), pm), -pm)
    last_formation = _add_months(_truncate(_as_date(sched["end"]), pm), -pm)
    if last_formation < first_formation:
        raise ValueError(
            f"backtest.schedule: end ({sched['end']}) precedes trade_start "
            f"({sched['trade_start']})."
        )

    regimes: list[Regime] = []
    start = first_formation
    while start <= last_formation:
        formation_end = min(_add_months(start, resel - pm), last_formation)
        regimes.append(
            Regime(
                index=len(regimes), cutoff=start,
                formation_start=start, formation_end=formation_end,
            )
        )
        start = _add_months(start, resel)
    return regimes
