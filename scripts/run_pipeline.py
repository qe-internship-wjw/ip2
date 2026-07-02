"""End-to-end pipeline driver.

Wires the stages from notes/notes.txt together so the modular pieces compose:

    1. Data        load -> join (point-in-time) -> accounting restate -> preprocess
    2. Universe    market set vs sector set; subindustry labels
    3. Factors     compute style + non-style scores
    4. Validation  neutralize -> single-factor tests -> reduce redundancy
    5. Portfolio   expected returns -> risk model -> optimize
    6. Backtest    walk-forward evaluation + metrics

Run from the repo root, e.g.:

    python -m scripts.run_pipeline --config config.yaml
"""

from __future__ import annotations

import argparse

from src import config as config_mod


def parse_args():
    parser = argparse.ArgumentParser(description="Run the financials factor strategy pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = config_mod.load(args.config)

    # 1. Data + universe (two lazy panels)
    # raw = loaders.load_all(cfg)
    # market_frame = preprocess.clean(joins.build_market_frame(raw, cfg), cfg)  # full universe, light
    # sector_panel = preprocess.clean(joins.build_sector_panel(raw, cfg), cfg)  # tradeable, rich
    #   build_market_frame feeds the cap-weighted factor-mimicking portfolios
    #   (Market/Country over all stocks) and carries a `tradeable` flag so loadings
    #   attach only to the ~2k tradeable names; build_sector_panel pre-filters to
    #   tradeable before the fundamentals as-of join.

    # 2. Factors: non-style loadings on market_frame; style scores on sector_panel
    # 3. Validation -> 4. Portfolio -> 5. Backtest
    # ...

    raise NotImplementedError("Pipeline stages not yet implemented.")


if __name__ == "__main__":
    main()
