"""Point-in-time feature selection over the ``data.cache`` artifacts.

Runs the three-step selection (single-factor gates -> correlation-cluster
representatives -> lasso; ``src/validation/selection.py``) as of one or more
cutoff formation periods, with no data past the cutoff entering any statistic
(DYNAMIC_SELECTION_PLAN.md). Consumes the precomputed
``neu_{pm}m`` / ``loadings_{pm}m`` / ``returns_{pm}m`` feathers -- run
``scripts/build_processed.py`` first.

Outputs (default ``<data.cache>/selection/``):

    scorecard_<cutoff>.feather   per-factor gate/cluster/lasso scorecard
    shortlists.json              {cutoff: [selected shorthands...]}
    selection_manifest.json      config hash, window, cutoffs (written last)

Cutoffs come from ``--cutoff`` (repeatable), ``--schedule`` (the regime
cutoffs implied by ``backtest.schedule``) or ``--full-sample`` (no cutoff --
the in-sample demo mode the tour notebook uses). Run inside the ``ip2`` conda
env, from the repo root:

    conda run -n ip2 python scripts/select_features.py --schedule
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.backtest.schedule import regime_schedule  # noqa: E402
from src.validation.selection import select_features  # noqa: E402


def _log(msg: str) -> None:
    print(f"[select_features] {msg}", flush=True)


def _write(df: pl.DataFrame, path: Path) -> None:
    """Atomic feather write (tmp + replace) so a crash never leaves a torn file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_ipc(tmp)
    tmp.replace(path)


def load_artifacts(cfg, cache: Path):
    """The three selection inputs at the rebalancing frequency."""
    pm = int(cfg["backtest"]["rebalancing_frequency_months"])
    frames = {}
    for name in (f"neu_{pm}m", f"loadings_{pm}m", f"returns_{pm}m"):
        path = cache / f"{name}.feather"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing -- run scripts/build_processed.py first."
            )
        frames[name] = pl.read_ipc(path)
    return frames[f"neu_{pm}m"], frames[f"loadings_{pm}m"], frames[f"returns_{pm}m"]


def run_selection(cfg, cutoffs, out_dir: Path, config_path: str) -> dict:
    """Select at each cutoff, write scorecards + shortlists, return the summary."""
    cache = Path(cfg["data"].get("cache", "data/processed"))
    if not cache.is_absolute():
        cache = PROJECT_ROOT / cache
    neu, loadings, returns = load_artifacts(cfg, cache)
    train_start = (cfg.get("backtest", {}).get("schedule") or {}).get("train_start")
    min_warn = int(
        (cfg.get("validation", {}).get("selection") or {}).get("min_shortlist_warn", 0)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    shortlists: dict[str, list[str]] = {}
    for cutoff in cutoffs:
        label = cutoff.isoformat() if cutoff is not None else "full"
        t0 = time.perf_counter()
        res = select_features(
            neu, returns, loadings, cfg, cutoff=cutoff, train_start=train_start
        )
        sc = res.scorecard
        n_gate = sc.filter(pl.col("single_pass")).height
        n_multi = sum(len(c) > 1 for c in res.clusters)
        _log(
            f"cutoff={label}: {sc.height} factors -> gate {n_gate} "
            f"(clusters>1: {n_multi}) -> reps {len(res.representatives)} "
            f"-> shortlist {len(res.shortlist)} in {time.perf_counter() - t0:.1f}s"
        )
        if min_warn and len(res.shortlist) < min_warn:
            _log(
                f"WARNING: cutoff={label} shortlist has {len(res.shortlist)} "
                f"< {min_warn} factors"
            )
        _log(f"  shortlist: {res.shortlist}")
        _write(sc, out_dir / f"scorecard_{label}.feather")
        shortlists[label] = res.shortlist

    (out_dir / "shortlists.json").write_text(json.dumps(shortlists, indent=2))
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config_path,
        "config_sha256": hashlib.sha256(
            json.dumps(cfg.raw, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "train_start": str(train_start),
        "cutoffs": list(shortlists),
    }
    # Written last: its presence marks a complete, consistent run.
    (out_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))
    return shortlists


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Point-in-time three-step factor selection on the processed artifacts."
    )
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument(
        "--out", default=None,
        help="output directory (default: <data.cache>/selection)",
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--cutoff", action="append", metavar="YYYY-MM-DD",
        help="selection cutoff formation period (repeatable)",
    )
    group.add_argument(
        "--schedule", action="store_true",
        help="use the regime cutoffs implied by backtest.schedule",
    )
    group.add_argument(
        "--full-sample", action="store_true",
        help="no cutoff (in-sample demo mode)",
    )
    args = ap.parse_args(argv)

    # Categorical stock_id columns are joined across independently-read feathers.
    pl.enable_string_cache()

    cfg = config.load(args.config)
    if args.full_sample:
        cutoffs = [None]
    elif args.cutoff:
        cutoffs = [datetime.fromisoformat(c).date() for c in args.cutoff]
    else:
        regimes = regime_schedule(cfg)
        cutoffs = [r.cutoff for r in regimes]
        _log(f"{len(regimes)} regime cutoffs: {[str(c) for c in cutoffs]}")

    out_dir = Path(args.out) if args.out else None
    if out_dir is None:
        cache = Path(cfg["data"].get("cache", "data/processed"))
        if not cache.is_absolute():
            cache = PROJECT_ROOT / cache
        out_dir = cache / "selection"

    t0 = time.perf_counter()
    run_selection(cfg, cutoffs, out_dir, args.config)
    _log(f"done in {time.perf_counter() - t0:,.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
