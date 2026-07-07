"""Transaction-cost model (src/portfolio/transaction_cost.py)."""

import numpy as np
import polars as pl
import pytest

from src.portfolio.transaction_cost import (
    MCAP_FLOOR,
    cost,
    free_float_mcap,
    linear_cost_coefficients,
)


def test_coefficient_formula_and_monotonicity():
    m = np.array([1e8, 1e9, 1e10])
    c = linear_cost_coefficients(m)
    expected = (3.0 * (11.0 / np.log10(m)) ** 6 + 3.0) * 1e-4
    assert np.allclose(c, expected)
    # Bigger float trades cheaper.
    assert c[0] > c[1] > c[2]


def test_floor_caps_the_coefficient():
    cap = linear_cost_coefficients([MCAP_FLOOR])[0]
    c = linear_cost_coefficients([1e3, MCAP_FLOOR, np.nan, np.inf])
    assert np.allclose(c, cap)


def test_cost_sums_abs_turnover():
    m = np.array([1e9, 1e9])
    c = linear_cost_coefficients(m)[0]
    assert cost([0.01, -0.02], m) == pytest.approx(c * 0.03)
    assert cost([0.0, 0.0], m) == 0.0


def test_free_float_percentage_scale_and_group_fallback():
    df = pl.DataFrame(
        {
            "stock_id": ["A", "B", "C", "D"],
            "industry": ["bank", "bank", "insurance_life", "insurance_life"],
            "mcap_usd": [1e9] * 4,
            "free_float_percentage": [50.0, None, 80.0, None],  # vendor 0-100 scale
        }
    )
    out = free_float_mcap(df, by="industry")
    vals = dict(zip(out["stock_id"].to_list(), out["free_float_mcap"].to_list()))
    assert vals["A"] == pytest.approx(0.5e9)  # scale auto-detected (median 65 > 1)
    assert vals["B"] == pytest.approx(0.5e9)  # null -> bank median fraction
    assert vals["C"] == pytest.approx(0.8e9)
    assert vals["D"] == pytest.approx(0.8e9)  # null -> insurance median fraction


def test_free_float_fraction_scale_and_full_float_default():
    df = pl.DataFrame(
        {
            "stock_id": ["A", "B"],
            "mcap_usd": [1e9, 2e9],
            "free_float_percentage": [0.5, None],  # already a fraction
        }
    )
    out = free_float_mcap(df)
    assert out["free_float_mcap"].to_list() == pytest.approx([0.5e9, 1e9])

    all_null = df.with_columns(free_float_percentage=pl.lit(None, dtype=pl.Float64))
    out = free_float_mcap(all_null)
    # No information anywhere -> full float, no haircut.
    assert out["free_float_mcap"].to_list() == pytest.approx([1e9, 2e9])


def test_free_float_lazy_in_lazy_out():
    df = pl.DataFrame(
        {"stock_id": ["A"], "mcap_usd": [1e9], "free_float_percentage": [0.4]}
    )
    out = free_float_mcap(df.lazy())
    assert isinstance(out, pl.LazyFrame)
    assert out.collect()["free_float_mcap"][0] == pytest.approx(0.4e9)
