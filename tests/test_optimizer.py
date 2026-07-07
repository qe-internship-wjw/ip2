"""MVO optimizer with L1 transaction costs (src/portfolio/optimizer.py).

Solver note: CLARABEL (cvxpy's default) is blocked by the machine's Application
Control policy, so tests pin OSQP explicitly -- the same path production takes.
"""

import numpy as np
import polars as pl
import pytest

from src.config import Config
from src.portfolio.optimizer import solve
from src.portfolio.risk_model import RiskModel

# Tight tolerances so analytic comparisons are meaningful on tiny problems.
OPTS = {"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iter": 200_000}


def _cfg(gross=2.0, net=(-0.2, 0.2), max_name=None, ra=2.0):
    return Config(
        raw={
            "portfolio": {
                "risk_aversion": ra,
                "constraints": {
                    "gross_leverage": gross,
                    "net_exposure": list(net),
                    "max_name_weight": max_name,
                },
                "optimizer": {"solver": "OSQP"},
            }
        }
    )


def _diag_model(d):
    n = len(d)
    return RiskModel(
        stock_ids=[f"S{i}" for i in range(n)],
        factors=["const"],
        B=np.zeros((n, 1)),
        F=np.zeros((1, 1)),
        D=np.asarray(d, dtype=float),
    )


def test_recovers_analytic_no_cost_solution():
    d = np.array([0.1, 0.2, 0.4])
    mu = np.array([0.02, 0.01, -0.01])
    w, diag = solve(
        mu, _diag_model(d), None, None, _cfg(gross=10, net=(-10, 10), ra=2.0),
        solver_opts=OPTS,
    )
    # Diagonal risk, no cost, slack constraints: w* = mu / (ra * D).
    assert diag["solved"]
    assert np.allclose(w["weight"].to_numpy(), mu / (2.0 * d), atol=1e-5)


def test_no_trade_region():
    d = np.array([0.04, 0.04])
    w_prev = np.array([0.05, -0.05])
    mu = 2.0 * d * w_prev  # alpha-risk gradient vanishes exactly at w_prev
    mff = np.array([1e7, 1e7])  # ~48 bps per unit turnover
    w, diag = solve(mu, _diag_model(d), w_prev, mff, _cfg(), solver_opts=OPTS)
    # Any trade costs more than it earns: the L1 kink holds the book still.
    assert np.allclose(w["weight"].to_numpy(), w_prev, atol=1e-6)
    assert diag["turnover"] == pytest.approx(0.0, abs=1e-6)


def test_gross_leverage_binds():
    w, diag = solve(
        np.array([0.5, 0.5]), _diag_model([0.01, 0.01]), None, None,
        _cfg(gross=1.0, net=(-2, 2)), solver_opts=OPTS,
    )
    assert diag["gross"] == pytest.approx(1.0, rel=1e-4)


def test_net_exposure_binds():
    w, diag = solve(
        np.array([0.5, 0.4]), _diag_model([0.01, 0.01]), None, None,
        _cfg(gross=2.0, net=(0.0, 0.0)), solver_opts=OPTS,
    )
    assert diag["net"] == pytest.approx(0.0, abs=1e-6)


def test_max_name_weight_bound():
    w, _ = solve(
        np.array([0.5, 0.5]), _diag_model([0.01, 0.01]), None, None,
        _cfg(gross=10, net=(-10, 10), max_name=0.1), solver_opts=OPTS,
    )
    assert np.all(np.abs(w["weight"].to_numpy()) <= 0.1 + 1e-6)


def test_extra_constraint_callable():
    w, _ = solve(
        np.array([0.5, 0.5]), _diag_model([0.01, 0.01]), None, None,
        _cfg(gross=1.0, net=(-2, 2)), constraints=[lambda w: w[0] == 0],
        solver_opts=OPTS,
    )
    assert abs(w["weight"][0]) < 1e-7


def test_sane_rejects_garbage_iterates():
    from src.portfolio.optimizer import _sane

    assert _sane(np.array([0.5, -0.5]), gross=2.0, net_lo=-0.2, net_hi=0.2)
    # The observed failure mode: an "optimal_inaccurate" iterate violating the
    # gross cap by orders of magnitude.
    assert not _sane(np.array([1e17, -1e17]), gross=2.0, net_lo=-0.2, net_hi=0.2)
    assert not _sane(np.array([np.nan, 0.0]), gross=2.0, net_lo=-0.2, net_hi=0.2)
    assert not _sane(None, gross=2.0, net_lo=-0.2, net_hi=0.2)
    # Loose convergence within the 5% slack is fine.
    assert _sane(np.array([1.02, -1.02]), gross=2.0, net_lo=-0.2, net_hi=0.2)


def test_infeasible_falls_back_to_w_prev():
    w_prev = np.array([0.02, 0.03])
    w, diag = solve(
        np.array([0.1, 0.1]), _diag_model([0.1, 0.1]), w_prev, None,
        _cfg(gross=0.1, net=(0.5, 0.5)),  # sum w = 0.5 impossible under gross 0.1
        solver_opts=OPTS,
    )
    assert not diag["solved"]
    assert np.allclose(w["weight"].to_numpy(), w_prev)


def test_factor_risk_enters_the_objective():
    # Two names with identical mu/D but one loads a risky factor: it gets less.
    rm = RiskModel(
        stock_ids=["S0", "S1"],
        factors=["f", "const"],
        B=np.array([[1.0, 1.0], [0.0, 1.0]]),
        F=np.diag([0.5, 0.0]),
        D=np.array([0.05, 0.05]),
    )
    w, diag = solve(
        np.array([0.02, 0.02]), rm, None, None, _cfg(gross=10, net=(-10, 10)),
        solver_opts=OPTS,
    )
    weights = w["weight"].to_numpy()
    assert diag["solved"]
    assert weights[0] < weights[1]
    assert diag["factor_risk"] > 0
