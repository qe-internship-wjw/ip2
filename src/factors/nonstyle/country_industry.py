"""Country and (sub)industry factors via factor-mimicking portfolios.

Following Bekaert et al. (2009), exposures are non-unit and dynamic rather than
0/1 dummies, derived from regression on factor-mimicking portfolios that use the
true country and industry labels of each stock:

    R_i = a * b_{i,Global} + sum_c gamma_c * b_{i,c} + sum_j beta_j * b_{i,j} + eps_i

where the first sum is country exposures and the second is industry terms.
Granularity (country grouping, subindustry depth) is a hyperparameter.
"""

from __future__ import annotations


def country_exposures(panel, cfg):
    """Pure country factor returns (zero-sum constrained cross-sectional fit)."""
    raise NotImplementedError


def industry_exposures(panel, cfg):
    """Hierarchical sector / bank-vs-insurance / life-vs-P&C factor loadings."""
    raise NotImplementedError
