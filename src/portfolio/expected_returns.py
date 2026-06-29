"""Expected-return estimation and factor integration.

Per-factor expected returns depend on the factor's kind:

    systematic:   mu_k   = beta_k * z_k          (priced risk exposure)
    behavioural:  mu_k,i = IC_k * sigma_i * z_k,i (mispricing scaled by idio vol)

Factors are combined by the *integration* approach: a strategic
weight vector alpha_k aggregates the per-factor expected returns,

    mu = sum_k alpha_k * mu_k

Default alpha_k = 1/K (equal); alternatives (e.g. IR-weighting) are configurable.
"""

from __future__ import annotations


def systematic_mu(beta, z):
    """Expected return for a systematic factor: mu_k = beta_k * z_k."""
    raise NotImplementedError


def behavioural_mu(ic, sigma, z):
    """Expected return for a behavioural factor: mu = IC * sigma * z."""
    raise NotImplementedError


def strategic_weights(factors, cfg):
    """Derive alpha_k (default equal 1/K; optionally IR-weighted)."""
    raise NotImplementedError


def integrate(per_factor_mu, alpha):
    """Aggregate per-factor expected returns: mu = sum_k alpha_k * mu_k."""
    raise NotImplementedError
