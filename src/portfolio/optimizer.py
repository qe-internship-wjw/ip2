"""Mean-variance optimizer.

Single-period MVO with an L1 transaction-cost penalty, over one rebalance
cross-section aligned to ``RiskModel.stock_ids``:

    max_w  mu' w  -  (lambda_ra / 2) (||F^0.5 B' w||^2 + sum_i D_i w_i^2)  -  c' |w - w_prev|
    s.t.   sum_i |w_i| <= gross_leverage
           net_lo <= sum_i w_i <= net_hi
           [optional] |w_i| <= max_name_weight

The quadratic uses the **structured** factor form (a K-dim norm plus a diagonal),
never the N x N Sigma. The L1 term induces the economically meaningful *no-trade
region*: when the marginal alpha of a trade is below its cost, the position stays
put. No neutrality constraints: market/sector beta is allowed in the book and the
broad-market beta is assumed hedged externally (accounting-only, see metrics).

Solver note: cvxpy's default (CLARABEL) is **blocked on this machine** by an
Application Control policy (its DLL will not load), so solvers are chosen
explicitly -- ``portfolio.optimizer.solver`` first if set, then every other
usable one of OSQP / SCS / HIGHS as a retry chain. The problem is
QP-representable (abs/norm1 reduce to linear constraints), which OSQP handles
natively; a solution is accepted only if it is finite and near-feasible
(:func:`_sane` -- ``optimal_inaccurate`` iterates can be garbage), otherwise the
next solver runs, and if none succeeds the book is carried unchanged.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .transaction_cost import linear_cost_coefficients

_SOLVER_ORDER = ("OSQP", "SCS", "HIGHS")


def _solver_chain(cfg):
    """Configured solver first, then every other usable one as a retry chain."""
    import cvxpy as cp

    want = cfg.get("portfolio", {}).get("optimizer", {}).get("solver")
    installed = [s for s in _SOLVER_ORDER if s in set(cp.installed_solvers())]
    if want:
        if want not in set(cp.installed_solvers()):
            raise ValueError(
                f"configured solver '{want}' is not usable here "
                f"(installed: {sorted(cp.installed_solvers())})"
            )
        return [want] + [s for s in installed if s != want]
    if not installed:
        raise RuntimeError(
            f"no usable cvxpy solver found (installed: {sorted(cp.installed_solvers())})"
        )
    return installed


def _sane(w_val, gross, net_lo, net_hi):
    """Reject non-finite or wildly infeasible iterates.

    A solver may report ``optimal_inaccurate`` yet hand back an unconverged
    iterate that violates the constraints by orders of magnitude (observed with
    OSQP on some real cross-sections); accepting it would poison a whole
    walk-forward run. Allow a generous 5% feasibility slack -- this guards
    against garbage, not loose convergence.
    """
    if w_val is None or not np.all(np.isfinite(w_val)):
        return False
    slack = max(1e-6, 0.05 * gross)
    net = float(np.sum(w_val))
    return (
        float(np.sum(np.abs(w_val))) <= gross + slack
        and net_lo - slack <= net <= net_hi + slack
    )


def _factor_half(F):
    """``H`` with ``H'H = F`` via eigendecomposition (F is PSD up to noise)."""
    evals, evecs = np.linalg.eigh((F + F.T) / 2.0)
    return np.diag(np.sqrt(np.clip(evals, 0.0, None))) @ evecs.T


def solve(mu, risk_model, w_prev, free_float_mcap, cfg, constraints=None,
          solver_opts=None):
    """Solve the MVO problem for target weights, net of transaction cost.

    Parameters
    ----------
    mu : (N,) expected returns aligned to ``risk_model.stock_ids``
        (rebalance-period units, like the risk model).
    risk_model : :class:`src.portfolio.risk_model.RiskModel`.
    w_prev : (N,) drifted previous weights aligned the same way (0 for entrants);
        ``None`` means an empty book. Names *leaving* the universe are the
        engine's concern (exit accounting), not the solver's.
    free_float_mcap : (N,) free-float mcap driving the per-name cost coefficients;
        ``None`` disables the cost term (diagnostics only).
    constraints : optional iterable of callables ``w -> constraint(s)`` -- the
        extension seam for per-name/liquidity bounds.
    solver_opts : extra solver kwargs (e.g. tight OSQP tolerances in tests).

    Returns ``(weights, diagnostics)``: a ``[stock_id, weight]`` frame and a dict
    with the solver status and the objective decomposition. On solver failure the
    book is carried unchanged (``w = w_prev``) with the failure recorded -- a long
    walk-forward run must not crash on one bad cross-section.
    """
    import cvxpy as cp

    mu = np.asarray(mu, dtype=float)
    n = mu.shape[0]
    if n != len(risk_model.stock_ids):
        raise ValueError("mu is not aligned to risk_model.stock_ids.")
    w_prev = np.zeros(n) if w_prev is None else np.asarray(w_prev, dtype=float)
    c = (
        np.zeros(n)
        if free_float_mcap is None
        else linear_cost_coefficients(free_float_mcap)
    )

    pcfg = cfg.get("portfolio", {}) or {}
    ccfg = pcfg.get("constraints", {}) or {}
    risk_aversion = float(pcfg.get("risk_aversion", 2.0))
    gross = float(ccfg.get("gross_leverage", 2.0))
    net_lo, net_hi = (float(b) for b in ccfg.get("net_exposure", [-0.2, 0.2]))
    max_name = ccfg.get("max_name_weight")

    half = _factor_half(risk_model.F)
    d = np.clip(np.asarray(risk_model.D, dtype=float), 0.0, None)

    w = cp.Variable(n)
    risk = cp.sum_squares(half @ (risk_model.B.T @ w)) + cp.sum(
        cp.multiply(d, cp.square(w))
    )
    objective = mu @ w - 0.5 * risk_aversion * risk - c @ cp.abs(w - w_prev)
    cons = [cp.norm1(w) <= gross, cp.sum(w) >= net_lo, cp.sum(w) <= net_hi]
    if max_name is not None:
        cons.append(cp.abs(w) <= float(max_name))
    for fn in constraints or ():
        extra = fn(w)
        cons.extend(extra if isinstance(extra, (list, tuple)) else [extra])

    problem = cp.Problem(cp.Maximize(objective), cons)
    chain = _solver_chain(cfg)
    solver, status, ok = chain[0], "solver_error", False
    w_val = None
    for i, s in enumerate(chain):
        # Solver-specific opts (e.g. OSQP tolerances) only apply to the
        # configured/primary solver; retries run on their own defaults.
        opts = solver_opts if i == 0 else None
        try:
            problem.solve(solver=s, **(opts or {}))
        except cp.error.SolverError:
            continue
        solver, status = s, problem.status
        if status in ("optimal", "optimal_inaccurate") and _sane(
            w.value, gross, net_lo, net_hi
        ):
            ok = True
            w_val = np.asarray(w.value, dtype=float)
            break
        if status in ("infeasible", "unbounded"):
            break  # a genuinely bad problem will not improve with another solver

    if not ok:
        w_val = w_prev.copy()

    dw = w_val - w_prev
    factor_risk = float(np.sum((half @ (risk_model.B.T @ w_val)) ** 2))
    idio_risk = float(np.sum(d * w_val**2))
    diagnostics = {
        "status": status,
        "solver": solver,
        "solved": ok,
        "alpha": float(mu @ w_val),
        "risk": factor_risk + idio_risk,
        "factor_risk": factor_risk,
        "idio_risk": idio_risk,
        "tc": float(c @ np.abs(dw)),
        "objective": float(
            mu @ w_val
            - 0.5 * risk_aversion * (factor_risk + idio_risk)
            - c @ np.abs(dw)
        ),
        "turnover": float(np.sum(np.abs(dw))),
        "gross": float(np.sum(np.abs(w_val))),
        "net": float(np.sum(w_val)),
    }
    weights = pl.DataFrame({"stock_id": risk_model.stock_ids, "weight": w_val})
    return weights, diagnostics
