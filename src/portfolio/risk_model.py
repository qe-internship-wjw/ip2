"""Covariance estimation: structural factor risk model + shrinkage.

The covariance is EMA-smoothed (coefficient > 0.9).
Use Ledoit-Wolf linear shrinkage to baseline as implemented in sklearn.
"""

from __future__ import annotations


def shrink(sample_cov, target, intensity=None):
    """Convex combination of sample covariance and structural target (Ledoit-Wolf)."""
    raise NotImplementedError


def ema_update(prev_cov, new_cov, coef):
    """Exponentially smooth the covariance estimate (coef > 0.9)."""
    raise NotImplementedError


def estimate(panel, factors, cfg):
    """Full covariance estimate: structural target -> shrink -> EMA."""
    raise NotImplementedError
