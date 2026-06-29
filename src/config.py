"""
Typed configuration loading and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Validated, in-memory view of config.yaml."""

    raw: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)


def load(path: str | Path = "config.yaml") -> Config:
    """Load and validate the YAML config into a Config object."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    validate(raw)
    return Config(raw=raw)


def validate(raw: dict[str, Any]) -> None:
    """Check required sections / value ranges. Raise on misconfiguration."""
    # TODO: assert presence of data/universe/factors/portfolio/backtest sections
    # and range-check e.g. risk_aversion, covariance EMA coefficient (>0.9).
    return None
