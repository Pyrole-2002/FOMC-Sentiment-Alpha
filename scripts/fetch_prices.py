"""Phase 1, step 1 -- download SPY prices and validate the trading calendar.

Run with:  uv run python scripts/fetch_prices.py
           uv run python scripts/fetch_prices.py --refresh   (re-download)

Prints a summary and the NYSE cross-check result. An empty discrepancy table is
the pass condition for this step's Definition of Done.
"""

from __future__ import annotations

import argparse

from src.config import load_config
from src.prices import download_prices, summarise, validate_against_nyse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true", help="re-download even if the cache exists"
    )
    args = parser.parse_args()

    cfg = load_config()
    print(f"Ticker           : {cfg.data.ticker}")
    print(f"auto_adjust      : {cfg.data.auto_adjust}  (all four OHLC columns adjusted)")
    print(f"Entry/exit field : {cfg.data.entry_price_field} -> {cfg.data.exit_price_field}")
    print()

    print("Downloading (or loading cache)...")
    prices = download_prices(cfg, force_refresh=args.refresh)

    print("\n--- Series summary ---")
    for key, value in summarise(prices).items():
        print(f"  {key:22s}: {value}")

    print("\n--- First 3 sessions ---")
    print(prices.head(3).to_string())
    print("\n--- Last 3 sessions ---")
    print(prices.tail(3).to_string())

    print("\n--- NYSE calendar cross-check ---")
    discrepancies = validate_against_nyse(prices)
    if discrepancies.empty:
        print("  PASS: price index matches the official NYSE session list exactly.")
        return 0

    print(f"  {len(discrepancies)} discrepancies found:\n")
    counts = discrepancies["issue"].value_counts()
    for issue, n in counts.items():
        print(f"    {issue:24s}: {n}")
    print("\n  First 25:")
    print(discrepancies.head(25).to_string(index=False))
    print(
        "\n  NOTE: 'missing_from_prices' is the dangerous direction -- searchsorted\n"
        "  would skip to the following session and shift an entry date by one."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
