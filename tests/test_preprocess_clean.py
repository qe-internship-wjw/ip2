"""Return data-quality scrub in :func:`src.data.preprocess.clean`.

Covers the raw-return gates from RETURNS_DATA_QUALITY.md: the USD price floor
(vanishingly small marks blanked so they never anchor a return), the return
magnitude band, and the max-gap guard on the null-imputation fallback.
"""

import datetime as dt

import polars as pl
import pytest

from src.data.preprocess import clean

CFG = {
    "preprocess": {
        "min_price_usd": 0.10,
        "max_abs_daily_return": 1.0,
        "max_impute_gap_days": 10,
    }
}


def _row(stock_id, date, price_local, total_return, *, fx=1.0, price_return=None, rf=0.0):
    return {
        "stock_id": stock_id,
        "date": date,
        "price_local": price_local,
        "fx_to_usd": fx,
        "total_return": total_return,
        "price_return": price_return,
        "risk_free_rate": rf,
    }


def _clean(rows, cfg=CFG):
    return clean(pl.DataFrame(rows), cfg).sort("stock_id", "date")


D = dt.date


def test_stub_round_trip_booked_flat():
    """A near-zero excursion (drop → stale → recovery) is fully neutralised."""
    rows = [
        _row("S", D(2020, 1, 1), 10.0, None),           # first row -> 0
        _row("S", D(2020, 1, 2), 10.5, 0.05),           # real +5%
        _row("S", D(2020, 1, 3), 0.005, -0.9995),       # drop INTO sub-floor stub
        _row("S", D(2020, 1, 4), 0.005, 0.0),           # stale stub
        _row("S", D(2020, 1, 7), 10.6, 2119.0),         # vendor "recovery" off stub
        _row("S", D(2020, 1, 8), 11.0, None),           # normal resumes (impute off 10.6)
    ]
    out = _clean(rows)
    tr = out["total_return"].to_list()
    assert tr[0] == pytest.approx(0.0)      # first row
    assert tr[1] == pytest.approx(0.05)     # real move kept
    assert tr[2] == pytest.approx(0.0)      # drop-into-stub gated flat
    assert tr[3] == pytest.approx(0.0)      # stale stub flat
    assert tr[4] == pytest.approx(0.0)      # recovery-off-stub gated flat
    assert tr[5] == pytest.approx(11.0 / 10.6 - 1.0)   # imputed off the last valid mark
    # The two stub marks are blanked; both stub-spanning returns are gated.
    assert out["_px_blanked"].sum() == 2
    assert out["_ret_gated"].sum() == 3     # rows 2,3,4 (drop, stale, recovery)


def test_band_gate_nulls_impossible_vendor_return():
    """A finite but impossible |return| > cap between valid prices is booked flat."""
    rows = [
        _row("S", D(2020, 1, 1), 10.0, None),
        _row("S", D(2020, 1, 2), 10.0, 5.0),    # +500% between two valid $10 marks
        _row("S", D(2020, 1, 3), 10.0, 0.5),    # +50% -- legitimate, kept
    ]
    out = _clean(rows)
    tr = out["total_return"].to_list()
    assert tr[1] == pytest.approx(0.0)
    assert tr[2] == pytest.approx(0.5)
    assert out["_px_blanked"].sum() == 0
    assert out["_ret_gated"].sum() == 1


def test_max_gap_guard_blocks_wide_imputation():
    """Imputing a null return across a > max_gap span books flat, not a stale jump."""
    rows = [
        _row("W", D(2020, 1, 1), 10.0, None),
        _row("W", D(2020, 1, 20), 20.0, None),   # 19-day gap > 10 -> NOT imputed
        _row("N", D(2020, 1, 1), 10.0, None),
        _row("N", D(2020, 1, 6), 12.0, None),    # 5-day gap <= 10 -> imputed
    ]
    out = _clean(rows)
    wide = out.filter(pl.col("stock_id") == "W")["total_return"].to_list()
    near = out.filter(pl.col("stock_id") == "N")["total_return"].to_list()
    assert wide[1] == pytest.approx(0.0)                 # gap too wide -> flat
    assert near[1] == pytest.approx(12.0 / 10.0 - 1.0)   # within gap -> imputed


def test_usd_floor_uses_fx_not_local_price():
    """The floor is judged in USD: a healthy local price can still be a sub-cent mark."""
    rows = [
        _row("S", D(2020, 1, 1), 100.0, None, fx=1.0),      # $100 valid
        _row("S", D(2020, 1, 2), 100.0, 0.0, fx=0.0005),    # $0.05 USD -> blanked
        _row("S", D(2020, 1, 3), 110.0, None, fx=1.0),      # valid; prev was stub
    ]
    out = _clean(rows)
    assert out["_px_blanked"].to_list() == [False, True, False]
    tr = out["total_return"].to_list()
    assert tr[1] == pytest.approx(0.0)   # stub -> flat
    assert tr[2] == pytest.approx(0.0)   # cannot impute off a blanked previous mark


def test_null_risk_free_rows_dropped_and_excess_recomputed():
    rows = [
        _row("S", D(2020, 1, 1), 10.0, 0.02, rf=0.001),
        _row("S", D(2020, 1, 2), 10.0, 0.03, rf=None),   # dropped
    ]
    out = _clean(rows)
    assert out.height == 1
    assert out["excess_return"][0] == pytest.approx(0.02 - 0.001)


def test_gates_disabled_when_unconfigured():
    """With no gate keys, clean falls back to plain impute/settle (legacy behaviour)."""
    cfg = {"preprocess": {}}
    rows = [
        _row("S", D(2020, 1, 1), 0.005, None),
        _row("S", D(2020, 1, 2), 10.0, 500.0),   # huge vendor return passes through
    ]
    out = _clean(rows, cfg)
    assert out["total_return"].to_list()[1] == pytest.approx(500.0)
    assert out["_px_blanked"].sum() == 0
    assert out["_ret_gated"].sum() == 0
