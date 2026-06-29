"""Universe definition and data splits.

Central to this project: the strategy is restricted to financials. We distinguish
the *market set* (all securities) from the *sector set* (banks + insurance), and
tag subindustries (retail vs investment banks; life vs P&C insurers) using GICS,
SIC and FactSet codes.

The market set is still needed to estimate market / country / industry non-style
factors; the sector set is where the strategy trades. Fine splits by region /
country / subindustry are executed implicitly via country- and industry-specific
factors rather than hard partitions, so this module produces membership masks and
subindustry labels rather than separate frames.
"""

from __future__ import annotations


def market_set(panel, cfg):
    """All securities used for estimating market/country/industry factors."""
    raise NotImplementedError


def sector_set(panel, cfg):
    """Securities restricted to banks and insurance (the tradable universe)."""
    raise NotImplementedError


def subindustry_labels(panel, cfg):
    """Reconcile GICS/SIC/FactSet into bank vs insurance and life vs P&C labels."""
    raise NotImplementedError
