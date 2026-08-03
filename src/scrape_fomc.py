"""Phase 1 -- crawl federalreserve.gov for FOMC policy statements.

Design note: **document URLs are never hardcoded.** Statement URL conventions
changed three times since 1994::

    1994-2005   /fomc/{YYYYMMDD}default.htm
    2006-2018   /newsevents/press/monetary/{YYYYMMDD}a.htm
    2019-now    /newsevents/pressreleases/monetary{YYYYMMDD}a.htm

Instead we crawl the *index* pages (``fomccalendars.htm`` for 2021+, and
``fomchistorical{YYYY}.htm`` for 1994-2020) and harvest their anchor hrefs.
Index-page structure is far more stable than document-URL conventions, so this
is robust to a format drift that a pattern-based scraper would silently miss.

The two timestamps (PLAN.md section 3.3) -- the distinction the whole project
rests on:

``event_date``
    Calendar date the *content* refers to: the decision day, which for a
    two-day meeting is day 2. (Verified: the Feb 3-4 1994 meeting is filed
    under ``19940204``.) **For display and joining only.**

``release_datetime``
    Timezone-aware ``America/New_York`` timestamp at which the text became
    public. **The only timestamp permitted to drive a trading decision.**
    For scheduled statements this is ``event_date`` at 14:00 ET; for the
    handful of unscheduled meetings it comes from
    ``config.scrape.release_time_overrides`` and is tagged
    ``release_time_source="manual_override"``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config


def fetch(url: str, cfg: Config) -> bytes:
    """GET ``url`` politely, with retries, returning the raw response body.

    Sets the configured User-Agent, sleeps ``request_delay_seconds`` between
    calls, and retries ``max_retries`` times with exponential backoff. Returns
    *bytes*, not text, so the sha256 recorded in the manifest is of exactly
    what the server sent (encoding guesses would make hashes unstable).
    """
    raise NotImplementedError("Phase 1")


def discover_documents(cfg: Config) -> pd.DataFrame:
    """Crawl the index pages and return one row per discoverable document.

    Returns
    -------
    DataFrame with columns:
        ``event_date`` (date), ``doc_type`` ("statement"), ``url`` (str).
        No text is downloaded here -- discovery is separated from fetching so
        the crawl plan can be inspected before hitting ~250 document URLs.
    """
    raise NotImplementedError("Phase 1")


def download_documents(docs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Fetch each document to ``data/raw/`` and append to the provenance manifest.

    ``data/raw`` is **immutable**: files are written once and never edited. The
    manifest (``data/raw/manifest.csv``) records, per file, the source URL, HTTP
    status, byte count, sha256 of the bytes, and UTC fetch time. This makes
    "raw is immutable" *verifiable* (see :func:`verify_manifest`) rather than
    merely aspirational, and lets a third party confirm their scrape matches
    ours byte-for-byte without us shipping the bytes.

    Already-downloaded URLs present in the manifest are skipped, so the crawl is
    resumable and the site is hit exactly once per document, ever.
    """
    raise NotImplementedError("Phase 1")


def parse_statement(html: bytes) -> str:
    """Extract the statement body text from a raw Fed HTML page.

    Parsed defensively: the Fed's page templates differ substantially across
    the three URL eras, so this selects the main content region by trying a
    sequence of selectors and falls back to whole-page text extraction rather
    than raising. A document that yields suspiciously little text is flagged,
    not silently dropped -- a silently empty statement would score as neutral
    and quietly dilute the signal.
    """
    raise NotImplementedError("Phase 1")


def build_documents_table(cfg: Config) -> pd.DataFrame:
    """Produce ``data/interim/documents.parquet``.

    Columns: ``event_date``, ``doc_type``, ``url``, ``raw_path``, ``text``,
    ``n_chars``, ``release_datetime`` (tz-aware America/New_York),
    ``release_time_source`` ("scheduled_1400ET" | "manual_override").
    """
    raise NotImplementedError("Phase 1")


def verify_manifest(manifest_path: Path) -> pd.DataFrame:
    """Re-hash every file in the manifest and report any mismatch.

    Proves the raw corpus has not drifted since it was scraped.
    """
    raise NotImplementedError("Phase 1")
