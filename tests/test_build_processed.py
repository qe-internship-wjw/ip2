"""Testable helpers of scripts/build_processed.py.

The heavy stages (raw loading, factor computes) need real data; these tests
cover the pure assembly logic: the dual-variant returns panel, the market /
benchmark series, and the ``input_frame`` routing the script relies on.
"""

import datetime as dt

import polars as pl
import pytest

from scripts.build_processed import META_COLS, market_series, returns_panel
from src.factors.base import registry

CFG = {"preprocess": {"winsorize_limits": [0.0, 0.5], "group_by": "period"}}


def _sector_panel():
    """Six stocks, one quarter, three prints each; stock S5 has one huge print."""
    dates = [dt.date(2020, 1, 15), dt.date(2020, 2, 14), dt.date(2020, 3, 16)]
    rows = []
    for i in range(6):
        for d in dates:
            spike = i == 5 and d == dates[1]
            rows.append(
                {
                    "stock_id": f"S{i}",
                    "date": d,
                    "excess_return": 5.0 if spike else 0.01 * i,
                    "industry": "bank" if i % 2 == 0 else "insurance_life",
                    "mcap_usd": 100.0 * (i + 1),
                    "free_float_percentage": 0.5,
                }
            )
    return pl.DataFrame(rows).lazy()


def test_returns_panel_schema_and_variants():
    out = returns_panel(_sector_panel(), None, CFG, period_months=3)
    assert out.columns == [
        "stock_id", "date", "period", *META_COLS, "ret_raw", "ret_wins",
    ]
    assert out["period"].unique().to_list() == [dt.date(2020, 1, 1)]
    assert out["date"].unique().to_list() == [dt.date(2020, 3, 16)]

    spike = out.filter(pl.col("stock_id") == "S5")
    clean = out.filter(pl.col("stock_id") == "S1")
    # The outlier print is clipped before compounding in ret_wins only.
    assert spike["ret_wins"][0] < spike["ret_raw"][0]
    # A mid-distribution stock is untouched by the daily clip.
    assert clean["ret_wins"][0] == pytest.approx(clean["ret_raw"][0])
    assert clean["ret_raw"][0] == pytest.approx(1.01**3 - 1.0)
    # Meta columns are period-end levels.
    assert clean["mcap_usd"][0] == 200.0
    assert clean["industry"][0] == "insurance_life"


def test_market_series_full_vs_tradeable():
    mf = pl.DataFrame(
        {
            "stock_id": ["A", "B", "C"] * 2,
            "date": [dt.date(2020, 1, 2)] * 3 + [dt.date(2020, 1, 3)] * 3,
            "excess_return": [0.01, 0.02, 0.10, 0.01, 0.02, 0.10],
            "mcap_usd": [100.0, 300.0, 600.0] * 2,
            "tradeable": [True, True, False] * 2,
        }
    ).lazy()
    out = market_series(mf).collect().sort("date")
    # mkt spans the full universe; bench only the tradeable names.
    mkt_expected = (0.01 * 100 + 0.02 * 300 + 0.10 * 600) / 1000.0
    bench_expected = (0.01 * 100 + 0.02 * 300) / 400.0
    assert out.columns == ["date", "mkt", "bench"]
    assert out["mkt"][0] == pytest.approx(mkt_expected)
    assert out["bench"][0] == pytest.approx(bench_expected)


def test_input_frame_routing():
    reg = registry()
    market_fed = {
        "MKT", "CTRY", "IND",
        "Beta~MKT", "Beta~CTRY", "Beta~IND (B)", "Beta~IND (I)",
    }
    for shorthand, cls in reg.items():
        expected = "market_frame" if shorthand in market_fed else "sector_panel"
        assert cls.input_frame == expected, shorthand
