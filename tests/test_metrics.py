"""Performance metrics (src/backtest/metrics.py)."""

import numpy as np
import pytest

from src.backtest.metrics import (
    excess_return,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
)


def test_excess_return_rf_and_hedge():
    out = excess_return([0.02], risk_free=[0.005], market_beta=[0.5],
                        market_returns=[0.02])
    assert out[0] == pytest.approx(0.02 - 0.005 - 0.5 * 0.02)

    # Hedge terms must come together.
    with pytest.raises(ValueError):
        excess_return([0.02], market_beta=[0.5])
    # Pass-through when nothing to subtract.
    assert excess_return([0.02])[0] == pytest.approx(0.02)


def test_max_drawdown():
    # Curve 1.1 -> 0.55 -> 0.66: trough is 50% below the 1.1 peak.
    assert max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(-0.5)
    assert max_drawdown([0.1, 0.2]) == pytest.approx(0.0)
    assert np.isnan(max_drawdown([]))


def test_sharpe_ratio_quarterly():
    r = [0.01, 0.03]
    expected = np.mean(r) / np.std(r, ddof=1) * 2.0  # sqrt(4) for quarterly
    assert sharpe_ratio(r, periods_per_year=4) == pytest.approx(expected)
    assert np.isnan(sharpe_ratio([0.02, 0.02], periods_per_year=4))  # zero vol
    assert np.isnan(sharpe_ratio([0.02], periods_per_year=4))  # too short


def test_information_ratio_is_sharpe_of_active():
    r, b = [0.02, 0.04], [0.01, 0.01]
    active = np.subtract(r, b)
    expected = active.mean() / active.std(ddof=1) * 2.0
    assert information_ratio(r, b, periods_per_year=4) == pytest.approx(expected)
