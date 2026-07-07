"""Structured risk model + shrinkage (src/portfolio/risk_model.py)."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.config import Config
from src.portfolio.risk_model import (
    ema_update,
    estimate,
    factor_returns,
    shrink_factor_cov,
    shrink_idio,
)


def _cfg(rebal=3, ema=0.9):
    return Config(
        raw={
            "portfolio": {
                "risk_model": {
                    "frequency_months": 1,
                    "factor_cov_window_months": 24,
                    "idio_window_months": 24,
                    "idio_min_obs": 6,
                },
                "covariance": {"ema_coefficient": ema},
            },
            "backtest": {"rebalancing_frequency_months": rebal},
        }
    )


def _months(n):
    return [dt.date(2020 + m // 12, m % 12 + 1, 1) for m in range(n)]


def test_factor_returns_recover_known_coefficients():
    rng = np.random.RandomState(0)
    rows = []
    for p in _months(2):
        for i in range(8):
            x1, x2 = float(rng.randn()), float(rng.randn())
            rows.append(
                {"stock_id": f"S{i}", "period": p, "x1": x1, "x2": x2,
                 "_y": 2.0 * x1 - 1.0 * x2 + 0.5}
            )
    rets, resid = factor_returns(pl.DataFrame(rows), ["x1", "x2"], "_y")
    for p in _months(2):
        d = {r["factor"]: r["f"] for r in rets.filter(pl.col("period") == p).to_dicts()}
        assert d["x1"] == pytest.approx(2.0, abs=1e-8)
        assert d["x2"] == pytest.approx(-1.0, abs=1e-8)
        assert d["const"] == pytest.approx(0.5, abs=1e-8)
    assert resid["resid"].abs().max() < 1e-8


def test_shrink_factor_cov_hits_the_diagonal_target():
    rng = np.random.RandomState(1)
    t, k = 40, 4
    common = rng.randn(t, 1)
    x = 0.5 * common + rng.randn(t, k)
    f, delta = shrink_factor_cov(x)

    assert 0.0 <= delta <= 1.0
    # Diagonal: sample variances, untouched.
    assert np.allclose(np.diag(f), x.var(axis=0, ddof=0))
    # Off-diagonals: exactly (1 - delta) * sample covariance -- shrunk toward 0.
    xc = x - x.mean(axis=0, keepdims=True)
    s = (xc.T @ xc) / t
    off = ~np.eye(k, dtype=bool)
    assert np.allclose(f[off], (1.0 - delta) * s[off])
    # PSD.
    assert np.linalg.eigvalsh((f + f.T) / 2).min() > -1e-10


def test_shrink_factor_cov_degenerate_inputs():
    f, delta = shrink_factor_cov(np.ones((1, 3)))  # T < 2: variances only
    assert np.allclose(f, 0.0) and delta == 1.0
    x = np.random.RandomState(2).randn(30, 3)
    x[:, 1] = 5.0  # zero-variance column keeps a zero row/col
    f, _ = shrink_factor_cov(x)
    assert np.allclose(f[1, :], 0.0) and np.allclose(f[:, 1], 0.0)


def test_shrink_idio_volatility_floor():
    s2 = np.array([1e-6, 0.01, 0.02, 0.5])
    out, delta = shrink_idio(s2, np.full(4, 24))
    assert 0.0 < delta <= 1.0
    assert out.mean() == pytest.approx(s2.mean())  # grand mean preserved
    assert out[0] > s2[0]  # near-zero variance floored up
    assert out[-1] < s2[-1]  # extreme variance pulled down

    same, d2 = shrink_idio([0.02, 0.02, 0.02], [24, 24, 24])
    assert d2 == 1.0 and np.allclose(same, 0.02)  # zero dispersion -> the mean


def test_ema_update_algebra():
    assert np.allclose(ema_update(None, [[1.0]], 0.9), [[1.0]])
    out = ema_update(np.array([[2.0]]), np.array([[1.0]]), 0.9)
    assert out[0, 0] == pytest.approx(0.9 * 2.0 + 0.1 * 1.0)


def _history(n_stocks=6, n_periods=24, seed=3):
    rng = np.random.RandomState(seed)
    periods = _months(n_periods)
    rets = pl.DataFrame(
        [
            {"period": p, "factor": f, "f": float(rng.randn() * 0.02)}
            for p in periods
            for f in ("x1", "x2", "const")
        ]
    )
    resid = pl.DataFrame(
        [
            {"stock_id": f"S{i}", "period": p, "resid": float(rng.randn() * 0.05)}
            for p in periods
            for i in range(n_stocks)
        ]
    )
    exposures = pl.DataFrame(
        {
            "stock_id": [f"S{i}" for i in range(n_stocks)],
            "industry": ["bank"] * (n_stocks // 2) + ["insurance_life"] * (n_stocks - n_stocks // 2),
            "x1": rng.randn(n_stocks),
            "x2": rng.randn(n_stocks),
        }
    )
    return exposures, rets, resid


def test_estimate_shapes_scaling_and_state():
    exposures, rets, resid = _history()
    rm = estimate(exposures, rets, resid, _cfg(rebal=3))

    assert rm.factors == ["x1", "x2", "const"]
    assert rm.B.shape == (6, 3) and np.allclose(rm.B[:, 2], 1.0)  # const appended
    assert rm.F.shape == (3, 3) and rm.D.shape == (6,)
    assert (rm.D > 0).all()
    assert np.allclose(rm.sigma_idio, np.sqrt(rm.D))
    sigma = rm.covariance()
    assert np.allclose(sigma, sigma.T)
    assert np.linalg.eigvalsh(sigma).min() > -1e-10

    # Monthly estimates scale to rebalance-period units: quarterly = 3 x monthly.
    rm_m = estimate(exposures, rets, resid, _cfg(rebal=1))
    assert np.allclose(rm.F, 3.0 * rm_m.F)
    assert np.allclose(rm.D, 3.0 * rm_m.D)

    # EMA of identical inputs is a fixed point; state threads without drift.
    rm2 = estimate(exposures, rets, resid, _cfg(rebal=3), state=rm.state)
    assert np.allclose(rm2.F, rm.F)
    assert np.allclose(rm2.D, rm.D)


def test_estimate_imputes_names_without_residual_history():
    exposures, rets, resid = _history()
    entrant = pl.DataFrame(
        {"stock_id": ["NEW"], "industry": ["bank"], "x1": [0.1], "x2": [-0.2]}
    )
    rm = estimate(pl.concat([exposures, entrant]), rets, resid, _cfg())
    assert rm.meta["n_idio_imputed"] == 1
    d = dict(zip(rm.stock_ids, rm.D))
    assert d["NEW"] > 0  # sub-universe median, shrunk -- never zero or null
