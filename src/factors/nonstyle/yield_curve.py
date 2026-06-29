"""Yield-curve (interest-rate risk) factors via Nelson-Siegel.

Equity sensitivities to level/slope changes are estimated by rolling
regression (e.g. 36-month). The decay factor tau is a
hyperparameter fixed during monthly OLS term-structure estimation.
"""

from __future__ import annotations


def fit_nelson_siegel(zero_curve, cfg):
    """Estimate level/slope/curvature per date per sovereign curve."""
    raise NotImplementedError


def yield_level_sensitivity(panel, ns_params, cfg):
    """Rolling sensitivity of equity returns to sovereign level changes."""
    raise NotImplementedError


def yield_slope_sensitivity(panel, ns_params, cfg):
    """Rolling sensitivity of equity returns to sovereign slope changes."""
    raise NotImplementedError
