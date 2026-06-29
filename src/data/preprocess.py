"""
Regularization: cleaning, winsorization, standardization.
"""

from __future__ import annotations


def clean(panel, cfg):
    """Repair null and erroneous raw values prior to factor generation."""
    raise NotImplementedError


def winsorize(scores, limits, by=None):
    """Clip cross-sectional outliers to the given quantile limits."""
    raise NotImplementedError


def fill_missing(scores, by=None):
    """Fill missing factor scores (e.g. cross-sectional median)."""
    raise NotImplementedError


def standardize(scores, by=None):
    """Cross-sectional z-score (optionally within group), the factor's z_k."""
    raise NotImplementedError


def regularize(scores, cfg):
    """Run winsorize -> fill_missing -> standardize on raw factor scores."""
    raise NotImplementedError
