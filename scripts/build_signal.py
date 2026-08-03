"""Phase 3 -- build the trailing Z-score signal and positions.

Run with:  uv run python scripts/build_signal.py

Augments `panel.parquet` with S, mu, sigma, Z, position and turnover.

⚠️ This script deliberately reports NOTHING about returns. Phase 3 produces the
signal; Phase 4 evaluates it. Looking at performance now would undermine the
pre-registration that makes the Phase 4 result trustworthy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.alpha_signal import build_signal, signal_sign, summarise_signal
from src.config import load_config


def main() -> int:
    cfg = load_config()
    pd.set_option("display.width", 200)

    sign = signal_sign(cfg)
    print("--- Pre-registered configuration ---")
    print(f"  aggregation      : {cfg.sentiment.aggregation}")
    print(f"  window L         : {cfg.signal.zscore_window_L} meetings")
    print(f"  threshold theta  : {cfg.signal.threshold_theta}")
    print(f"  hypothesis       : {cfg.signal.hypothesis_sign}  ->  sign = {sign:+d}")
    print(f"                     (positive Z {'LONG' if sign > 0 else 'SHORT'} SPY)")
    print()

    panel = build_signal(cfg)
    print("Build-time leak assertion PASSED (Z reproduced by an independent backward loop).\n")

    stats = summarise_signal(panel, cfg)

    print("--- Sample ---")
    print(f"  events                    : {stats['n_events']}")
    print(
        f"  warm-up rows (NaN Z)      : {stats['n_warmup_nan']}   "
        f"(expected {cfg.signal.zscore_window_L})"
    )
    print(f"  usable observations       : {stats['n_usable']}")

    print("\n--- Z distribution ---")
    print(
        f"  mean {stats['z_mean']:+.4f}   sd {stats['z_std']:.4f}   "
        f"min {stats['z_min']:+.3f}   max {stats['z_max']:+.3f}"
    )
    print("  (mean should sit near 0 and sd near 1 by construction; a large")
    print("   departure would mean the trailing window is badly mis-specified.)")

    valid = panel["Z"].notna()
    print("\n  Z histogram:")
    counts, edges = np.histogram(panel.loc[valid, "Z"], bins=12)
    for count, lo, hi in zip(counts, edges[:-1], edges[1:], strict=False):
        print(f"    [{lo:+5.2f}, {hi:+5.2f})  {'#' * int(count / max(counts) * 44):44s} {count}")

    print("\n--- Positions ---")
    print(f"  long  (+1) : {stats['n_long']:3d}")
    print(f"  short (-1) : {stats['n_short']:3d}")
    print(f"  flat   (0) : {stats['n_flat']:3d}   (includes {stats['n_warmup_nan']} warm-up rows)")
    print(f"  in market  : {stats['pct_in_market']}% of events")
    print(
        f"  turnover   : {stats['total_turnover']:.0f} total, "
        f"{stats['mean_turnover_per_event']} per event"
    )

    if stats["n_long"] + stats["n_short"] == 0:
        print(
            "\n  ⚠️  The rule NEVER trades. That is a finding, not a bug: at "
            f"theta={cfg.signal.threshold_theta} no surprise was ever large enough."
        )

    print("\n--- Warm-up rows (must all be NaN Z, position 0) ---")
    head = panel.head(cfg.signal.zscore_window_L + 2)[
        ["doc_date", "S", "mu", "sigma", "Z", "position"]
    ]
    print(head.to_string(index=False))

    print("\n--- Largest surprises (by |Z|) ---")
    top = panel.loc[valid].reindex(panel.loc[valid, "Z"].abs().sort_values(ascending=False).index)
    cols = ["doc_date", "is_scheduled", "S", "mu", "sigma", "Z", "position"]
    print(top.head(10)[cols].to_string(index=False))

    print("\n--- Signal by era (is the Z-score doing its job?) ---")
    era = panel.loc[valid].assign(year=lambda d: d["doc_date"].dt.year)
    bucket = pd.cut(
        era["year"],
        bins=[1999, 2007, 2015, 2021, 2027],
        labels=["2000-07", "2008-15", "2016-21", "2022-26"],
    )
    summary = era.groupby(bucket, observed=True).agg(
        n=("Z", "size"), S_mean=("S", "mean"), Z_mean=("Z", "mean"), Z_sd=("Z", "std")
    )
    print(summary.round(4).to_string())
    print("\n  S_mean still drifts by era (that is the raw non-stationarity).")
    print("  Z_mean should be much closer to 0 everywhere -- that is the")
    print("  normalisation working. If Z_mean still drifted, L would be too long.")

    out = cfg.paths.resolve("data_processed") / "panel.parquet"
    print(f"\nWrote {out}")
    print("\nNOTE: no return data was examined. Evaluation is Phase 4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
