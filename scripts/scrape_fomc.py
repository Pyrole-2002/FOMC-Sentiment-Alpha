"""Phase 1, step 2 -- discover, download and parse FOMC documents.

Run with:
    uv run python scripts/scrape_fomc.py --discover-only   # inspect the plan first
    uv run python scripts/scrape_fomc.py                   # discover + download + parse
    uv run python scripts/scrape_fomc.py --verify          # re-hash the raw corpus

Discovery is deliberately separate from downloading so the crawl plan can be
inspected before ~500 document URLs are requested from someone else's server.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import load_config
from src.scrape_fomc import (
    build_documents_table,
    discover_documents,
    download_documents,
    load_manifest,
    verify_manifest,
)


def _print_discovery(docs: pd.DataFrame) -> None:
    print(f"\nDiscovered {len(docs)} documents.\n")

    print("--- By type ---")
    print(docs["doc_type"].value_counts().to_string())

    statements = docs[docs["doc_type"] == "statement"].copy()
    statements["year"] = statements["doc_date"].map(lambda d: d.year)

    print("\n--- Statements per year ---")
    by_year = statements.groupby("year").size()
    for year, n in by_year.items():
        bar = "#" * n
        print(f"  {year}: {n:3d}  {bar}")

    print("\n--- Scheduled vs unscheduled (statements) ---")
    print(statements["is_scheduled"].value_counts(dropna=False).to_string())

    unscheduled = statements[statements["is_scheduled"] == False]  # noqa: E712
    if not unscheduled.empty:
        print(f"\n--- The {len(unscheduled)} unscheduled statements ---")
        for _, row in unscheduled.iterrows():
            print(f"  {row['doc_date']}  {str(row['meeting_heading'])[:58]}")

    unknown = statements[statements["is_scheduled"].isna()]
    if not unknown.empty:
        print(f"\n⚠️  {len(unknown)} statements have an UNRECOVERABLE heading:")
        for _, row in unknown.head(15).iterrows():
            print(f"  {row['doc_date']}  {row['url']}")

    print("\n--- URL era coverage (statements) ---")

    def era(url: str) -> str:
        if "/fomc/" in url:
            return "1994-1995  /fomc/*default.htm"
        if "/boarddocs/" in url:
            return "1997-2005  /boarddocs/press/*"
        if "/press/monetary/" in url:
            return "2006-2010  /newsevents/press/monetary/*"
        return "2011-now   /newsevents/pressreleases/*"

    print(statements["url"].map(era).value_counts().sort_index().to_string())


def _print_parse_report(table: pd.DataFrame, cfg) -> None:
    statements = table[table["doc_type"] == "statement"]

    n_total = len(statements)
    n_flagged = int(statements["is_flagged"].sum())
    n_parsed = n_total - n_flagged

    print("\n--- Parse report (statements) ---")
    print(f"  scraped : {n_total}")
    print(f"  parsed  : {n_parsed}")
    print(f"  flagged : {n_flagged}   (text < {cfg.scrape.min_text_chars} chars)")
    print(
        f"  identity: {n_total} == {n_parsed} + {n_flagged}  ->  "
        f"{'OK' if n_total == n_parsed + n_flagged else 'MISMATCH'}"
    )

    print("\n--- Text length distribution (chars) ---")
    print(statements["n_chars"].describe().round(0).to_string())

    print("\n--- Selector used (which template era matched) ---")
    print(statements["selector_used"].value_counts().to_string())

    flagged = statements[statements["is_flagged"]]
    if not flagged.empty:
        print(f"\n⚠️  {len(flagged)} FLAGGED documents (kept, never silently dropped):")
        for _, row in flagged.iterrows():
            print(f"  {row['doc_date']}  {row['n_chars']:6d} chars  {row['url']}")

    print("\n--- Release-time provenance ---")
    print(statements["release_time_source"].value_counts().to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="crawl index pages and print the plan; download nothing",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="re-parse the already-downloaded corpus; no network at all",
    )
    parser.add_argument(
        "--verify", action="store_true", help="re-hash the raw corpus against the manifest and exit"
    )
    args = parser.parse_args()

    cfg = load_config()
    discovery_cache = cfg.paths.resolve("data_interim") / "discovery.parquet"

    if args.parse_only:
        # Re-parsing must never re-hit the network: the raw corpus is immutable,
        # so improving the parser is a pure local operation over bytes we
        # already hold. This is exactly what the raw/interim split is for.
        if not discovery_cache.exists():
            print(f"No discovery cache at {discovery_cache}; run a full scrape first.")
            return 1
        docs = pd.read_parquet(discovery_cache)
        print(f"Re-parsing {len(docs)} already-downloaded documents (no network)...")
        table = build_documents_table(cfg, docs)
        _print_parse_report(table, cfg)
        return 0

    if args.verify:
        problems = verify_manifest(cfg)
        manifest = load_manifest(cfg)
        print(f"Manifest rows: {len(manifest)}")
        if problems.empty:
            print("PASS: every file re-hashes to its recorded sha256.")
            return 0
        print(f"FAIL: {len(problems)} problems\n")
        print(problems.to_string(index=False))
        return 1

    print(f"Scrape window   : {cfg.data.scrape_start_date} -> today")
    print(f"Analysis window : {cfg.data.start_date} -> today  (disclosure-regime change)")
    print(f"Document types  : {cfg.scrape.document_types}")
    print(
        f"Politeness      : {cfg.scrape.request_delay_seconds}s delay, "
        f"{cfg.scrape.max_retries} retries"
    )
    print(
        "\nCrawling index pages "
        f"({cfg.scrape.historical_year_start}-{cfg.scrape.historical_year_end} "
        "historical + modern calendar)..."
    )

    docs = discover_documents(cfg)
    _print_discovery(docs)

    if args.discover_only:
        print("\n--discover-only: stopping before download.")
        return 0

    print(f"\nDownloading {len(docs)} documents (already-fetched URLs are skipped)...")
    docs = download_documents(docs, cfg)

    # Cache the discovery result (now carrying raw_path) so --parse-only can
    # iterate on the parser without touching the Fed's servers again.
    discovery_cache.parent.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(discovery_cache)

    problems = verify_manifest(cfg)
    print(f"\nManifest integrity: {'PASS' if problems.empty else f'FAIL ({len(problems)})'}")

    print("\nParsing...")
    table = build_documents_table(cfg, docs)
    _print_parse_report(table, cfg)

    out = cfg.paths.resolve("data_interim") / "documents.parquet"
    print(f"\nWrote {out.relative_to(out.parent.parent.parent)}  ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
