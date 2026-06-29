"""Market factor: global market movement aggregated across the universe."""

from __future__ import annotations

from ..base import Applicability, Factor, FactorKind


class Market(Factor):
    name = "Market"
    shorthand = "MKT"
    sleeve = "NonStyle"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS

    def compute(self, panel, cfg):
        """Aggregate global market return from all stocks in the universe."""
        raise NotImplementedError
