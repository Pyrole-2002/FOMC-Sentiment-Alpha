"""Phase 1, step 3 -- align documents to tradable entries and build the panel.

Run with:  uv run python scripts/build_panel.py

Produces `data/processed/panel.parquet`: the Phase 1 deliverable. Prints the
funnel (so nothing can vanish silently), the leak-guard verification, the
emergency-meeting spot checks, and the event-overlap report.
"""

from __future__ import annotations

import pandas as pd

from src.align import build_panel, check_overlap, session_open_instants
from src.config import load_config
from src.prices import download_prices, trading_days

ET = "America/New_York"

SPOT_CHECKS = [
    ("2001-09-17", "reopening after 9/11 (14:00 default)"),
    ("2008-10-08", "coordinated emergency cut, 07:00 ET PRE-OPEN"),
    ("2008-12-16", "cut to the zero lower bound"),
    ("2020-03-15", "COVID cut to 0-0.25%, SUNDAY 17:00 ET"),
    ("2026-07-29", "a recent scheduled meeting"),
]


def main() -> int:
    cfg = load_config()
    pd.set_option("display.width", 200)

    print("Building panel...\n")
    panel = build_panel(cfg)
    counts = panel.attrs["counts"]

    print("--- Funnel (every drop is counted, none is silent) ---")
    labels = {
        "discovered": "documents discovered",
        "statements": "of which statements",
        "after_dropping_flagged": "after dropping flagged parses",
        "after_sample_start": f"after sample start >= {cfg.data.start_date}",
        "after_unscheduled_filter": f"after unscheduled filter "
        f"(include={cfg.sample.include_unscheduled})",
        "with_entry_date": "with a resolvable entry date",
    }
    prev = None
    for key, label in labels.items():
        n = counts[key]
        delta = "" if prev is None else f"   ({n - prev:+d})"
        print(f"  {label:46s}: {n:4d}{delta}")
        prev = n

    print(f"\n  ==> n = {len(panel)} events in the primary sample")
    print(
        f"      after the Z-score warm-up (L={cfg.signal.zscore_window_L}): "
        f"n = {len(panel) - cfg.signal.zscore_window_L}"
    )

    print("\n--- Composition ---")
    print(f"  scheduled   : {int((panel['is_scheduled'] == True).sum())}")  # noqa: E712
    print(f"  unscheduled : {int((panel['is_scheduled'] == False).sum())}")  # noqa: E712
    print(f"  date range  : {panel['doc_date'].min()} -> {panel['doc_date'].max()}")

    print("\n--- Leak guard ---")
    have = panel[panel["entry_date"].notna()]
    opens = session_open_instants(pd.DatetimeIndex(have["entry_date"]), cfg)
    releases = pd.to_datetime(have["release_datetime"]).dt.tz_convert(ET)
    # Align both to the same index before subtracting: subtracting a raw ndarray
    # from a DatetimeIndex falls back to slow object arithmetic and loses .dt.
    lag_hours = (pd.Series(opens, index=have.index) - releases).dt.total_seconds() / 3600

    print(
        f"  entry open > release for all {len(opens)} rows : "
        f"{'PASS' if (lag_hours > 0).all() else 'FAIL'}"
    )
    print(f"  smallest release->entry lag : {lag_hours.min():.2f} hours")
    print(f"  median  release->entry lag  : {lag_hours.median():.2f} hours")
    print(f"  largest release->entry lag  : {lag_hours.max():.2f} hours")

    print("\n--- Release-time provenance (in the primary sample) ---")
    print(panel["release_time_source"].value_counts().to_string())
    same_session = (
        pd.DatetimeIndex(panel["entry_date"]).date
        == pd.to_datetime(panel["release_datetime"]).dt.tz_convert(ET).dt.date
    )
    print(f"  pre-open releases entered SAME session: {int(same_session.sum())}")

    disagree = panel[
        (panel["release_time_source"] == "manual_override") & panel["parsed_release_time"].notna()
    ]
    if not disagree.empty:
        print("\n  Cross-check, hand-entered override vs. the document's own text:")
        for _, r in disagree.iterrows():
            stated = pd.Timestamp(r["release_datetime"]).tz_convert(ET).strftime("%H:%M")
            mark = "AGREE" if stated == str(r["parsed_release_time"])[:5] else "DISAGREE"
            print(
                f"    {r['doc_date']}  override {stated}  "
                f"document says {r['parsed_release_time']}  -> {mark}"
            )

    print("\n--- Spot checks (the cases naive date arithmetic gets wrong) ---")
    for iso, why in SPOT_CHECKS:
        row = panel[panel["doc_date"].astype(str) == iso]
        if row.empty:
            print(f"  {iso}  NOT IN SAMPLE   ({why})")
            continue
        r = row.iloc[0]
        rel = pd.Timestamp(r["release_datetime"]).tz_convert(ET)
        same = pd.Timestamp(r["entry_date"]).date() == rel.date()
        print(
            f"  {iso}  release {rel:%Y-%m-%d %H:%M %Z}  ->  entry "
            f"{pd.Timestamp(r['entry_date']).date()}"
            f"{'  [SAME session: pre-open release]' if same else ''}"
        )
        print(f"             {why}  [{r['release_time_source']}]")

    print("\n--- Event overlap (independence of forward returns) ---")
    sessions = trading_days(download_prices(cfg))
    report = check_overlap(panel["entry_date"], sessions, cfg.backtest.horizons_days)
    print(f"  events                    : {report['n_events']}")
    print(
        f"  gap between events (sessions): min {report['min_gap_sessions']}, "
        f"median {report['median_gap_sessions']:.0f}, max {report['max_gap_sessions']}"
    )
    for h, pairs in report["overlapping_pairs"].items():
        status = "clean" if not pairs else f"{len(pairs)} OVERLAPPING PAIRS"
        print(f"    h={h:2d}: {status}")
        for a, b, gap in pairs:
            print(f"          {a} -> {b}  ({gap} sessions apart)")

    print("\n--- Forward-return coverage ---")
    for h in cfg.backtest.horizons_days:
        col = f"fwd_ret_{h}"
        n_nan = int(panel[col].isna().sum())
        print(
            f"  {col:12s}: {len(panel) - n_nan:4d} present, {n_nan} NaN  "
            f"(mean {panel[col].mean() * 100:+.3f}%, sd {panel[col].std() * 100:.3f}%)"
        )

    out = cfg.paths.resolve("data_processed") / "panel.parquet"
    print(f"\nWrote {out}")
    print(f"Columns: {list(panel.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
