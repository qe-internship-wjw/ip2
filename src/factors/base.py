"""Factor base class and registry.

Every factor (style and non-style) is a `Factor` carrying the metadata the rest
of the pipeline needs to treat factors polymorphically:

* ``sleeve``        -- Value / Quality / Momentum / Low Vol / Leverage / etc.
* ``kind``          -- SYSTEMATIC vs BEHAVIOURAL.
                       selects the expected-return formula in portfolio.expected_returns
                       (systematic: mu_k = beta_k * z_k ; behavioural: mu = IC * sigma * z).
* ``applicability`` -- which subuniverse the factor is defined on
                       (all-financials / banks / insurance).
* ``neutralize``    -- whether the score is residualised against the non-style
                       design (validation.neutralization). False for the
                       structural-beta signals: they are *explicit* bets on that
                       design, and a stacked beta is exactly collinear with its
                       expanded ``beta_{g}`` block, so residualising would
                       annihilate the signal. Regularization still applies.

Concrete factors implement `compute()` to return raw cross-sectional scores;
neutralization and standardization happen downstream so factors stay declarative.
A simple registry lets the pipeline enumerate factors from config without
hard-coding them, which is the main extensibility seam.
"""

from __future__ import annotations

from enum import Enum


class FactorKind(Enum):
    SYSTEMATIC = "systematic"
    BEHAVIOURAL = "behavioural"


class Applicability(Enum):
    ALL_FINANCIALS = "all_financials"
    BANKS = "banks"
    INSURANCE = "insurance"


class Factor:
    """Declarative factor definition.

    Subclasses set the class attributes and implement `compute`.
    """

    name: str = ""
    shorthand: str = ""
    sleeve: str = ""
    kind: FactorKind = FactorKind.SYSTEMATIC
    applicability: Applicability = Applicability.ALL_FINANCIALS
    neutralize: bool = True

    def compute(self, panel, cfg):
        """Return raw (pre-neutralization) cross-sectional factor scores."""
        raise NotImplementedError


_REGISTRY: dict[str, type[Factor]] = {}


def register(cls: type[Factor]) -> type[Factor]:
    """Class decorator to add a Factor subclass to the global registry."""
    _REGISTRY[cls.shorthand or cls.__name__] = cls
    return cls


def registry() -> dict[str, type[Factor]]:
    """Return the registered factors keyed by shorthand (auto-populating first)."""
    from . import style  # noqa: F401  (registers the style factors)
    from .nonstyle import country_industry, market  # noqa: F401
    return dict(_REGISTRY)
