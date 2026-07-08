"""Precompute the downsampled pipeline artifacts into ``data.cache``.

Builds, from the raw feather tables and **without any country or time-window
filter**, everything feature selection and the backtest consume:

    returns_{1,3}m.feather   [stock_id, date, period, industry, mcap_usd,
                              free_float_percentage, ret_raw, ret_wins]
    scores_{1,3}m.feather    [stock_id, date, period, industry, <style shorthands>]
    loadings_{1,3}m.feather  [stock_id, date, period, MKT, beta_*, is_*]
    neu_{1,3}m.feather       [stock_id, period, industry, <neutralized shorthands>]
    delist_events.feather    [stock_id, last_active_date, delist_date, reason,
                              delist_return]
    market_daily.feather     [date, mkt, bench]
    manifest.json            build stamp, config hash, artifact inventory

Design notes
------------
* Cross-sectional statistics (winsorization bounds, medians, z-scores, OLS
  residuals) are computed on the **full unfiltered panel**. The raw ``scores_*``
  / ``loadings_*`` artifacts defer regularization so a sliced study can rebuild
  its own cross-sections; the ``neu_*`` convenience frames are only valid for
  full-universe analyses.
* ``ret_wins`` winsorizes the **daily** returns cross-sectionally per date
  (bounds from the full panel) before compounding; ``ret_raw`` compounds the
  untouched dailies. Both are delisting-trimmed and terminal-settled
  (DELISTING_HANDLING.md). Feature selection reads ``ret_wins``; the backtest's
  realized P&L reads ``ret_raw`` -- ``validation._common.forward_returns``
  dispatches on the saved panels directly, with ``target_col`` picking the
  variant.
* The market frame and sector panel are materialized once to intermediate
  feathers and re-scanned, so the ~20 factor computes don't re-execute the raw
  join graph.
* Frequencies default to the config's risk-model months (1) and rebalancing
  months (3); output dir defaults to ``data.cache``.

Run inside the ``ip2`` conda env, from the repo root:

    conda run -n ip2 python scripts/build_processed.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config, universe  # noqa: E402
from src.data import delisting, joins, loaders, preprocess  # noqa: E402
from src.factors.base import registry  # noqa: E402
from src.factors.nonstyle.market import market_returns  # noqa: E402
from src.validation import neutralization as nz  # noqa: E402
from src.validation._common import periodic_returns  # noqa: E402

# Period-end levels carried on the returns panels: `industry` keys the
# sub-universe splits, `mcap_usd` weights quantile portfolios, and
# `free_float_percentage` feeds transaction_cost.free_float_mcap post-slice.
META_COLS = ("industry", "mcap_usd", "free_float_percentage")


def _log(msg: str) -> None:
    print(f"[build_processed] {msg}", flush=True)


def _write(df: pl.DataFrame, path: Path) -> pl.DataFrame:
    """Atomic feather write (tmp + replace) so a crash never leaves a torn file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_ipc(tmp)
    tmp.replace(path)
    return df


def _materialize(lf: pl.LazyFrame, path: Path) -> pl.LazyFrame:
    """Persist a lazy frame and return a scan of it (one execution of the graph)."""
    try:
        lf.sink_ipc(path)
    except Exception:  # plan not streamable end-to-end -> collect, then write
        lf.collect(engine="streaming").write_ipc(path)
    return pl.scan_ipc(path)


# Transient audit columns emitted by preprocess.clean (RETURNS_DATA_QUALITY.md).
_SCRUB_AUDIT_COLS = ("_px_blanked", "_ret_gated")


def _log_scrub(scan: pl.LazyFrame, label: str) -> pl.LazyFrame:
    """Log how many rows each data-quality gate touched, then drop the audit cols.

    The counts are read straight from the already-materialized feather (a cheap
    columnar sum, no re-execution of the join graph), keeping the scrub auditable
    without a second pass.
    """
    counts = scan.select(
        n=pl.len(),
        n_price_blanked=pl.col("_px_blanked").sum(),
        n_return_gated=pl.col("_ret_gated").sum(),
    ).collect().row(0, named=True)
    _log(
        f"{label} scrub: blanked {counts['n_price_blanked']:,} sub-floor prices, "
        f"gated {counts['n_return_gated']:,} impossible returns "
        f"(of {counts['n']:,} rows)"
    )
    return scan.drop(_SCRUB_AUDIT_COLS)


def build_panels(cfg, tmp_dir: Path):
    """Cleaned full-universe market frame + sector panel (materialized) and
    the delisting events of the tradeable set, all unfiltered."""
    raw = loaders.load_all(cfg)
    market_frame = _materialize(
        preprocess.clean(joins.build_market_frame(raw, cfg), cfg),
        tmp_dir / "market_frame.feather",
    )
    market_frame = _log_scrub(market_frame, "market frame")
    _log("market frame materialized")
    sector_panel = _materialize(
        preprocess.clean(joins.build_sector_panel(raw, cfg), cfg),
        tmp_dir / "sector_panel.feather",
    )
    sector_panel = _log_scrub(sector_panel, "sector panel")
    _log("sector panel materialized")
    events = delisting.delist_events(
        raw["price"].join(universe.tradable_ids(raw, cfg), on="stock_id", how="semi"),
        cfg,
    )
    return market_frame, sector_panel, events


def _join_all(frames):
    return functools.reduce(
        lambda a, b: a.join(b, on=["stock_id", "date"], how="left"), frames
    )


def factor_frames(market_frame, sector_panel, cfg):
    """Daily style-score and non-style-loading frames (lazy).

    Factor inputs are routed by ``Factor.input_frame`` -- the structural-beta
    signals and the loadings read the market frame, everything else the sector
    panel. The style/non-style split (scores vs design) stays module-based.
    """
    reg = registry()
    panels = {"sector_panel": sector_panel, "market_frame": market_frame}
    styles = sorted(
        s for s, c in reg.items() if c.__module__.startswith("src.factors.style")
    )
    nonstyle = sorted(
        s for s, c in reg.items() if c.__module__.startswith("src.factors.nonstyle")
    )
    styles_daily = _join_all(
        [sector_panel.select("stock_id", "date", "industry")]
        + [reg[s]().compute(panels[reg[s].input_frame], cfg) for s in styles]
    )
    loads_daily = _join_all(
        [sector_panel.select("stock_id", "date")]
        + [reg[s]().compute(panels[reg[s].input_frame], cfg) for s in nonstyle]
    )
    return styles_daily, loads_daily


def returns_panel(sector_panel, events, cfg, period_months: int) -> pl.DataFrame:
    """Per-(security, period) compounded excess returns, raw and winsorized.

    Both variants run through the same trim/compound/settle pipeline
    (:func:`src.validation._common.periodic_returns`); they differ only in the
    daily winsorization. The meta levels ride on the winsorized pass (sampled at
    the period end, untouched).
    """
    daily = sector_panel.select("stock_id", "date", "excess_return", *META_COLS)
    wins = periodic_returns(
        daily,
        period_months=period_months,
        winsorize_limits=cfg["preprocess"]["winsorize_limits"],
        weight_col=META_COLS,
        delist_events=events,
    ).rename({"_ret": "ret_wins"})
    raw = periodic_returns(
        daily,
        period_months=period_months,
        winsorize_limits=None,
        delist_events=events,
    ).select("stock_id", "period", ret_raw=pl.col("_ret"))
    return wins.join(raw, on=["stock_id", "period"], how="inner").select(
        "stock_id", "date", "period", *META_COLS, "ret_raw", "ret_wins"
    )


def market_series(market_frame) -> pl.LazyFrame:
    """Daily cap-weighted series: `mkt` (full universe, the hedging factor) and
    `bench` (tradeable-universe benchmark)."""
    mkt = market_returns(market_frame).rename({"_factor": "mkt"})
    bench = (
        market_frame.filter(pl.col("tradeable"))
        .group_by("date")
        .agg(
            bench=(pl.col("excess_return") * pl.col("mcap_usd")).sum()
            / pl.col("mcap_usd").sum()
        )
    )
    return mkt.join(bench, on="date", how="left").sort("date")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Precompute downsampled returns / factor panels (unfiltered)."
    )
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument(
        "--out", default=None, help="output directory (default: config data.cache)"
    )
    ap.add_argument(
        "--period-months", type=int, nargs="+", default=None,
        help="downsampling frequencies in months (default: config risk-model "
        "frequency_months + rebalancing_frequency_months)",
    )
    ap.add_argument(
        "--keep-intermediate", action="store_true",
        help="keep the materialized market frame / sector panel feathers",
    )
    args = ap.parse_args(argv)

    # Categorical columns are joined across independently-scanned tables.
    pl.enable_string_cache()

    cfg = config.load(args.config)
    root = Path(cfg["data"]["root"])
    if not root.is_absolute():
        cfg.raw["data"]["root"] = str(PROJECT_ROOT / root)

    out_dir = Path(args.out) if args.out else Path(cfg["data"].get("cache", "data/processed"))
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_intermediate"
    tmp_dir.mkdir(exist_ok=True)

    pms = args.period_months or sorted(
        {
            int(cfg["portfolio"]["risk_model"]["frequency_months"]),
            int(cfg["backtest"]["rebalancing_frequency_months"]),
        }
    )

    t0 = time.perf_counter()
    artifacts: dict[str, dict] = {}

    def save(df: pl.DataFrame, name: str) -> None:
        _write(df, out_dir / name)
        artifacts[name] = {"rows": df.height, "columns": df.columns}
        _log(f"wrote {name}: {df.height:,} rows x {df.width} cols")

    _log("building cleaned panels (full universe, no country/window filters)...")
    market_frame, sector_panel, events = build_panels(cfg, tmp_dir)
    save(events, "delist_events.feather")

    styles_daily, loads_daily = factor_frames(market_frame, sector_panel, cfg)

    for pm in pms:
        _log(f"--- {pm}-month grid ---")
        grid = preprocess.rebalance_grid(sector_panel, cfg, period_months=pm)
        scores = preprocess.to_rebalance(styles_daily, grid).collect()
        save(scores, f"scores_{pm}m.feather")
        loads = preprocess.to_rebalance(loads_daily, grid).collect()
        save(loads, f"loadings_{pm}m.feather")
        save(returns_panel(sector_panel, events, cfg, pm), f"returns_{pm}m.feather")
        # Convenience frames: cross-sections of the FULL universe. A sliced study
        # must rebuild from scores_*/loadings_* (see module docstring).
        neu = nz.neutralize(
            preprocess.regularize(scores, cfg), loads, cfg,
            by=cfg["preprocess"].get("group_by", "date"),
        )
        save(neu, f"neu_{pm}m.feather")

    save(market_series(market_frame).collect(), "market_daily.feather")

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config": args.config,
        "config_sha256": hashlib.sha256(
            json.dumps(cfg.raw, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "period_months": pms,
        "filters": "none (full universe, full history)",
        "notes": {
            "ret_wins": "daily excess returns winsorized per date over the full "
            "panel, then compounded; delist-trimmed and terminal-settled",
            "ret_raw": "raw compounded; same trimming/settlement",
            "scores/loadings": "raw (pre-regularization); regularize/neutralize "
            "after any slicing",
            "neu": "regularized+neutralized on the FULL universe; sliced studies "
            "must rebuild from scores_*/loadings_*",
        },
        "artifacts": artifacts,
    }
    # Written last: its presence marks a complete, consistent build.
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if not args.keep_intermediate:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    _log(f"done in {time.perf_counter() - t0:,.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
