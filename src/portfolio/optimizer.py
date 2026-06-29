"""Mean-variance optimizer.

Single-period MVO with a turnover (transaction-cost) penalty:

    max_w  w' mu - (lambda / 2) w' Sigma w - TC(dw)

"""

from __future__ import annotations


def solve(mu, cov, w_prev, free_float_mcap, cfg, constraints=None):
    """Solve the MVO problem for target weights w, net of transaction cost."""
    raise NotImplementedError
