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

    # 1. Data
    # raw = loaders.load_all(cfg)
    # panel = joins.build_panel(raw, cfg)
    # panel = accounting.restate(panel, cfg)
    # panel = preprocess.clean(panel, cfg)

    # 2. Universe
    # market = universe.market_set(panel, cfg)
    # sector = universe.sector_set(panel, cfg)

    # 3. Factors -> 4. Validation -> 5. Portfolio -> 6. Backtest
    # ...

    raise NotImplementedError("Pipeline stages not yet implemented.")


if __name__ == "__main__":
    main()
