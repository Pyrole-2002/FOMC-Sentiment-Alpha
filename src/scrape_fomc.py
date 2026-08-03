"""Phase 1 -- crawl federalreserve.gov for FOMC policy statements (and minutes).

Design note: **document URLs are never constructed.** Verified empirically in
Phase 1, the statement URL convention has changed **four** times since 1994::

    1994-1995   /fomc/{YYYYMMDD}default.htm
    1997-2005   /boarddocs/press/{general|monetary}/{YYYY}/{YYYYMMDD}/
    2006-2010   /newsevents/press/monetary/{YYYYMMDD}a.htm
    2011-now    /newsevents/pressreleases/monetary{YYYYMMDD}a.htm

(PLAN.md originally described three eras; the ``boarddocs`` era, which holds
**61 statements across 1997-2005**, was missing. A pattern-based scraper would
have silently returned nothing for nine years of the sample.)

Instead we crawl the *index* pages and harvest their anchors. Coverage splits
cleanly, verified: ``fomchistorical{YYYY}.htm`` exists for 1994-2020 and 404s
from 2021; ``fomccalendars.htm`` covers 2021 onward. No gap, no overlap.

Identifying the policy statement
--------------------------------
The two index templates label their links differently:

*Historical pages* -- the policy statement's anchor text is exactly
``"Statement"``. Sibling documents are self-labelling and therefore trivially
excluded: ``"Implementation Note"``, ``"Statement on Longer-Run Goals and
Monetary Policy Strategy"``, ``"Balance Sheet Normalization Principles and
Plans"``.

*Modern calendar* -- the row reads ``Statement: PDF | HTML``, so the statement
is the ``"HTML"`` anchor at ``monetary{date}a.htm``. Note the ``a`` suffix is
load-bearing: ``a1`` is the Implementation Note and ``b``/``c`` are separate
policy documents.

⚠️ Do not filter the historical era on an ``a.htm`` suffix: the 2008-12-16
statement lives at ``20081216b.htm``. Anchor text is the reliable discriminator
there, URL suffix is the reliable one in the modern era, and the two templates
must each use their own.

Scheduled vs unscheduled
------------------------
Each index entry sits under a self-describing heading, which gives us the flag
for free::

    "January 29-30 Meeting - 2008"          -> scheduled
    "October 7 Conference Call - 2008"      -> unscheduled
    "March 15 (unscheduled) Meeting - 2020" -> unscheduled
    "March 23 (notation vote) - 2020"       -> unscheduled

The two timestamps (PLAN.md section 3.3)
----------------------------------------
``doc_date``
    The date embedded in the URL. Empirically this is the date the document was
    **published**, not the date of the meeting it describes: ``20081008a.htm``
    sits under "October 7 Conference Call" because the call was on the 7th and
    the statement went out on the 8th. This is the timestamp we want.

``release_datetime``
    ``doc_date`` combined with a time, as a timezone-aware ``America/New_York``
    instant. **The only timestamp permitted to drive a trading decision.**
    The time is the 14:00 ET default unless verified otherwise in
    ``config.scrape.release_time_overrides`` -- see the safety principle there:
    an unverified time must err *late*, because erring early is a leak.

``meeting_heading``
    The index heading, kept verbatim for provenance and for deriving
    ``is_scheduled``.
"""

from __future__ import annotations

import hashlib
import re
import time as time_module
from datetime import date, datetime, time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import Config

ET = ZoneInfo("America/New_York")

MANIFEST_COLUMNS = [
    "url",
    "raw_path",
    "http_status",
    "n_bytes",
    "sha256",
    "fetched_at_utc",
]

# Every URL shape that has ever hosted an FOMC policy statement. Used only to
# RECOGNISE harvested anchors -- never to build a URL.
_STATEMENT_URL_RE = re.compile(
    r"(/fomc/(?P<d1>\d{8})default\.htm"
    r"|/boarddocs/press/[a-z]+/\d{4}/(?P<d2>\d{8})/?(default\.htm)?$"
    r"|/newsevents/press/monetary/(?P<d3>\d{8})[a-z]\d*\.htm"
    r"|/newsevents/pressreleases/monetary(?P<d4>\d{8})[a-z]\d*\.htm)"
)
_MINUTES_URL_RE = re.compile(r"/monetarypolicy/fomcminutes(?P<d>\d{8})\.htm")

# Modern-calendar statements only: the `a` suffix, and NOT `a1` (Implementation
# Note) or `b`/`c` (separate policy documents).
_MODERN_STATEMENT_RE = re.compile(r"/newsevents/pressreleases/monetary(\d{8})a\.htm$")

_UNSCHEDULED_MARKERS = ("conference call", "unscheduled", "notation vote", "videoconference")


class FetchError(RuntimeError):
    """A document could not be retrieved after all retries."""


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch(url: str, cfg: Config, session: requests.Session | None = None) -> requests.Response:
    """GET ``url`` politely, with retries and exponential backoff.

    Returns the ``Response`` so callers can hash ``response.content`` -- the raw
    **bytes**, not decoded text. Decoding requires guessing an encoding, and a
    guess would make the sha256 unstable across runs, defeating the point of
    recording it.
    """
    sess = session or requests
    headers = {"User-Agent": cfg.scrape.user_agent}
    last_exc: Exception | None = None

    for attempt in range(cfg.scrape.max_retries + 1):
        try:
            response = sess.get(url, headers=headers, timeout=cfg.scrape.timeout_seconds)
            time_module.sleep(cfg.scrape.request_delay_seconds)
            if response.status_code == 200:
                return response
            # 4xx other than 429 will not improve with retrying.
            if 400 <= response.status_code < 500 and response.status_code != 429:
                return response
            last_exc = FetchError(f"HTTP {response.status_code} for {url}")
        except requests.RequestException as exc:
            last_exc = exc
        if attempt < cfg.scrape.max_retries:
            time_module.sleep(cfg.scrape.request_delay_seconds * (2**attempt))

    raise FetchError(
        f"failed to fetch {url} after {cfg.scrape.max_retries + 1} attempts: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _nearest_heading(anchor) -> str | None:
    """Recover the index heading that an anchor sits under.

    The Fed's index pages nest each meeting's links inside a Bootstrap panel
    whose preceding heading names the meeting. We walk up a bounded number of
    ancestors and take the nearest preceding heading element; bounded rather
    than unbounded so a template change degrades to ``None`` (which becomes a
    flag) instead of silently attaching a heading from a different meeting.
    """
    node = anchor
    for _ in range(8):
        node = node.parent
        if node is None:
            return None
        heading = node.find_previous(["h4", "h5", "h3"])
        if heading is not None:
            return " ".join(heading.get_text().split())
    return None


def _is_scheduled(heading: str | None) -> bool | None:
    """Classify a meeting heading. ``None`` means "no evidence either way".

    ⚠️ Returning ``None`` rather than ``False`` for an unrecognised heading is
    the whole point. An earlier version returned ``bool("meeting" in heading)``,
    which silently labelled every modern-calendar row *unscheduled* -- those
    rows read "January 26-27 Statement: PDF | HTML ...", with no literal word
    "Meeting". Absence of evidence became evidence of absence, and 61 routine
    meetings were misclassified without anything failing.

    Three-valued logic keeps "I don't know" distinguishable from "no", so the
    caller can apply a template-specific default and the report can surface
    whatever is still unknown.
    """
    if not heading:
        return None
    lowered = heading.lower()
    if any(marker in lowered for marker in _UNSCHEDULED_MARKERS):
        return False
    if "meeting" in lowered:
        return True
    return None


def _date_from_url(url: str) -> date | None:
    match = _STATEMENT_URL_RE.search(url) or _MINUTES_URL_RE.search(url)
    if match is None:
        return None
    raw = next((v for v in match.groupdict().values() if v), None)
    if raw is None:
        return None
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError:
        return None


def _discover_from_historical(html: str, base_url: str, year: int) -> list[dict]:
    """Harvest statements and minutes from a ``fomchistorical{YYYY}.htm`` page."""
    soup = BeautifulSoup(html, "lxml")
    found: list[dict] = []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        text = " ".join(anchor.get_text().split())

        if text == "Statement" and _STATEMENT_URL_RE.search(href):
            doc_type = "statement"
        elif _MINUTES_URL_RE.search(href):
            doc_type = "minutes"
        else:
            continue

        doc_date = _date_from_url(href)
        if doc_date is None:
            continue

        heading = _nearest_heading(anchor)
        found.append(
            {
                "doc_type": doc_type,
                "doc_date": doc_date,
                "url": urljoin(base_url, href),
                "anchor_text": text,
                "meeting_heading": heading,
                "is_scheduled": _is_scheduled(heading),
                "source_index": f"fomchistorical{year}.htm",
            }
        )
    return found


def _discover_from_calendar(html: str, base_url: str) -> list[dict]:
    """Harvest statements and minutes from ``fomccalendars.htm`` (2021 onward).

    Different template, different discriminator: here the statement is the
    ``HTML`` anchor whose URL carries the bare ``a`` suffix.
    """
    soup = BeautifulSoup(html, "lxml")
    found: list[dict] = []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        anchor_text = " ".join(anchor.get_text().split())

        # ⚠️ BOTH conditions are required. The `a` suffix alone is NOT sufficient:
        # the annual "Statement on Longer-Run Goals and Monetary Policy Strategy"
        # is published at monetary20250822a.htm and would otherwise be ingested
        # as a policy statement. On this template the row reads
        # "Statement: PDF | HTML", so the policy statement is the "HTML" anchor.
        if _MODERN_STATEMENT_RE.search(href) and anchor_text == "HTML":
            doc_type = "statement"
        elif _MINUTES_URL_RE.search(href):
            doc_type = "minutes"
        else:
            continue

        doc_date = _date_from_url(href)
        if doc_date is None:
            continue

        row = anchor.find_parent(class_=re.compile(r"fomc-meeting"))
        heading = " ".join(row.get_text().split())[:120] if row else None
        scheduled = _is_scheduled(heading)
        # Template-specific default. The modern calendar writes "January 27-28",
        # not "January 27-28 Meeting", so `_is_scheduled` legitimately returns
        # None for a perfectly ordinary meeting. Every row on this page IS a
        # scheduled meeting unless it carries an explicit marker -- which
        # `_is_scheduled` has already checked for and would have returned False.
        if scheduled is None and heading:
            scheduled = True

        found.append(
            {
                "doc_type": doc_type,
                "doc_date": doc_date,
                "url": urljoin(base_url, href),
                "anchor_text": anchor_text,
                "meeting_heading": heading,
                "is_scheduled": scheduled,
                "source_index": "fomccalendars.htm",
            }
        )
    return found


def discover_documents(cfg: Config, session: requests.Session | None = None) -> pd.DataFrame:
    """Crawl every index page and return one row per discoverable document.

    Downloads no documents -- discovery is deliberately separated from fetching
    so the crawl plan can be inspected before ~500 document URLs are requested.

    Returns columns: ``doc_type``, ``doc_date``, ``url``, ``anchor_text``,
    ``meeting_heading``, ``is_scheduled``, ``source_index``.
    """
    sess = session or requests.Session()
    rows: list[dict] = []

    for year in range(cfg.scrape.historical_year_start, cfg.scrape.historical_year_end + 1):
        url = cfg.scrape.historical_url_template.format(year=year)
        response = fetch(url, cfg, sess)
        if response.status_code != 200:
            # Expected for years the Fed has not archived this way; recorded by
            # the caller's summary rather than silently swallowed.
            continue
        rows.extend(_discover_from_historical(response.text, cfg.scrape.base_url, year))

    response = fetch(cfg.scrape.calendar_url, cfg, sess)
    if response.status_code == 200:
        rows.extend(_discover_from_calendar(response.text, cfg.scrape.base_url))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "discovery found no documents at all -- the index page templates have "
            "almost certainly changed. Inspect before trusting anything downstream."
        )

    # A document can legitimately appear on two index pages; keep one row.
    df = df.drop_duplicates(subset=["url"]).sort_values(["doc_type", "doc_date"])
    df = df[df["doc_type"].isin(cfg.scrape.document_types)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Downloading + provenance
# ---------------------------------------------------------------------------


def _raw_path_for(cfg: Config, row: pd.Series) -> Path:
    subdir = "fomc_statements" if row["doc_type"] == "statement" else "fomc_minutes"
    stem = row["doc_date"].strftime("%Y%m%d")
    # Disambiguate the rare case of two documents of one type on one date by
    # appending a short hash of the URL. Deterministic, so re-runs are stable.
    suffix = hashlib.sha256(row["url"].encode()).hexdigest()[:6]
    return cfg.paths.resolve("data_raw") / subdir / f"{stem}_{suffix}.htm"


def load_manifest(cfg: Config) -> pd.DataFrame:
    path = cfg.paths.resolve("data_raw") / "manifest.csv"
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return pd.read_csv(path)


def download_documents(
    docs: pd.DataFrame, cfg: Config, session: requests.Session | None = None
) -> pd.DataFrame:
    """Fetch each document to ``data/raw/`` and append to the provenance manifest.

    ``data/raw`` is **immutable**: files are written once and never edited. The
    manifest records, per file, the source URL, HTTP status, byte count, sha256
    of the raw bytes, and UTC fetch time. That makes "raw is immutable"
    *verifiable* (see :func:`verify_manifest`) rather than merely aspirational,
    and lets a third party confirm their scrape matches ours byte-for-byte
    without us shipping the bytes.

    URLs already present in the manifest are **skipped**, so the crawl is
    resumable and the Fed's server is hit exactly once per document, ever.
    """
    sess = session or requests.Session()
    manifest = load_manifest(cfg)
    already = set(manifest["url"]) if not manifest.empty else set()

    new_rows: list[dict] = []
    docs = docs.copy()
    docs["raw_path"] = [
        str(_raw_path_for(cfg, row).relative_to(cfg.paths.resolve("data_raw").parent.parent))
        for _, row in docs.iterrows()
    ]

    for _, row in docs.iterrows():
        if row["url"] in already:
            continue
        path = _raw_path_for(cfg, row)
        path.parent.mkdir(parents=True, exist_ok=True)

        response = fetch(row["url"], cfg, sess)
        payload = response.content
        if response.status_code == 200:
            path.write_bytes(payload)

        new_rows.append(
            {
                "url": row["url"],
                "raw_path": row["raw_path"],
                "http_status": response.status_code,
                "n_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "fetched_at_utc": datetime.now(tz=ZoneInfo("UTC")).isoformat(timespec="seconds"),
            }
        )

    if new_rows:
        manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
        manifest_path = cfg.paths.resolve("data_raw") / "manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)

    return docs


def verify_manifest(cfg: Config) -> pd.DataFrame:
    """Re-hash every file in the manifest and report mismatches.

    Proves the raw corpus has not drifted since it was scraped. An empty frame
    is the pass condition.
    """
    manifest = load_manifest(cfg)
    repo_root = cfg.paths.resolve("data_raw").parent.parent
    problems: list[dict] = []

    for _, row in manifest.iterrows():
        if int(row["http_status"]) != 200:
            continue
        path = repo_root / row["raw_path"]
        if not path.exists():
            problems.append({"raw_path": row["raw_path"], "issue": "file_missing"})
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != row["sha256"]:
            problems.append({"raw_path": row["raw_path"], "issue": "sha256_mismatch"})

    return pd.DataFrame(problems, columns=["raw_path", "issue"])


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Tried in order. The Fed's page templates differ substantially across the four
# URL eras, so we try the most specific container first and widen from there.
_CONTENT_SELECTORS = [
    "div#article",
    "div.col-xs-12.col-sm-8.col-md-8",
    "div#content",
    "div#leftText",
    "td.text",
    "div.col-xs-12",
]

_BOILERPLATE_PREFIXES = (
    "Board of Governors of the Federal Reserve System",
    "The Federal Reserve, the central bank of the United States",
    "Skip to main content",
    "FRB: Press Release",
    "For media inquiries",
    "Last update:",
    "Implementation Note issued",
    "Share",
)

# Site navigation and footer chrome. The old templates (1994-2005) have no
# semantic content container, so the whole-page fallback drags these in --
# "Home | Accessibility | Contact Us | Last update" and so on. They are
# sentiment-neutral filler that would dilute every aggregation in Phase 2, and
# they appear ONLY in the old era, which would bias those years relative to the
# modern ones. A systematic, era-correlated dilution is worse than noise: it is
# a confound.
_NAV_LINES = frozenset(
    {
        "home",
        "accessibility",
        "contact us",
        "press releases",
        "search",
        "submit",
        "site map",
        "a-z index",
        "home page",
        "news and",
        "events",
        "news and events",
        "monetary policy",
        "back to home",
        "share",
        "for immediate release",
        "skip to main content",
        "submit button",
    }
)

# Trailing breadcrumb on the 1997-2005 boarddocs template, e.g. "2002 Monetary
# policy". Matched by pattern rather than listed literally because the year
# varies. This is the only junk that survives nav-stripping on that era.
_NAV_PATTERNS = (re.compile(r"^\d{4}\s+Monetary policy$", re.IGNORECASE),)

# Phrases that mean "this is an error page", even when served with HTTP 200.
_SOFT_404_MARKERS = (
    "page not found",
    "the page you were looking for has been moved",
    "no longer exists",
)

# The documents state their own release time, e.g. "For release at 2:00 p.m. EDT"
# or "For release at 5:00 p.m. EDT" (the 2020-03-15 emergency cut). Parsing it
# turns the weakest link in the alignment chain -- an assumed timestamp -- into
# evidence taken from the artifact itself.
_RELEASE_TIME_RE = re.compile(
    r"for\s+release\s+at\s+(\d{1,2})[:.](\d{2})\s*([ap])\.?\s*m\.?",
    re.IGNORECASE,
)


def parse_statement(html: bytes) -> tuple[str, str]:
    """Extract statement body text from raw Fed HTML.

    Returns ``(text, selector_used)`` so the phase report can show which
    template era each document matched -- a cheap way to notice that one era is
    parsing differently from the rest.

    Parsed **defensively**: selectors are tried in order and we fall back to
    whole-page extraction rather than raising. A caller must still check the
    length against ``cfg.scrape.min_text_chars`` -- see the warning below.

    ⚠️ This function never decides that a document is bad. It returns whatever
    it found and lets the caller flag it. An empty statement scores as *neutral*
    in Phase 2 -- a perfectly plausible-looking number that quietly dilutes the
    signal without ever raising an error. Silent degradation is more dangerous
    than loud failure, so the length check must happen and must be reported.
    """
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    for selector in _CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if node is None:
            continue
        text = _clean_text(node.get_text(separator="\n"))
        if len(text) >= 400:  # plausible-statement floor for selector acceptance
            return text, selector

    return _clean_text(soup.get_text(separator="\n")), "whole_page_fallback"


def _clean_text(raw: str) -> str:
    """Normalise whitespace and strip site chrome.

    Note the two filters do different jobs. ``_BOILERPLATE_PREFIXES`` drops
    lines that *begin* with known filler (headers, media-contact footers).
    ``_NAV_LINES`` drops lines that *are exactly* a navigation label -- an exact
    match, deliberately, so a genuine sentence that merely contains the word
    "home" survives.
    """
    lines = [" ".join(line.split()) for line in raw.splitlines()]
    lines = [ln for ln in lines if ln]
    lines = [ln for ln in lines if not ln.startswith(_BOILERPLATE_PREFIXES)]
    lines = [ln for ln in lines if ln.strip(" |").lower() not in _NAV_LINES]
    lines = [ln for ln in lines if not any(p.match(ln.strip(" |")) for p in _NAV_PATTERNS)]
    lines = [ln for ln in lines if ln.strip(" |")]
    return "\n".join(lines).strip()


def looks_like_error_page(text: str) -> bool:
    """Detect a 'Page not Found' body served with HTTP 200 (a *soft* 404).

    ⚠️ Why this is not paranoia. A Fed 404 body renders to ~1,170 characters of
    prose -- comfortably ABOVE the 500-character flag threshold. A hard 404 is
    already safe (``download_documents`` never writes the file, so parsing sees
    a missing file and flags it). But a soft 404 would pass the length check,
    parse cleanly, score as roughly neutral in Phase 2, and silently dilute the
    signal. The length heuristic alone is not sufficient; content must be
    checked too.
    """
    head = text[:600].lower()
    return any(marker in head for marker in _SOFT_404_MARKERS)


def extract_release_time(text: str) -> time | None:
    """Parse the release time the document states about itself.

    FOMC statements carry a line such as ``For release at 2:00 p.m. EDT``; the
    2020-03-15 emergency cut says ``For release at 5:00 p.m. EDT``, which
    independently confirms the hand-entered override for that date.

    Returns ``None`` when the document says only "For immediate release" (the
    older convention) or states no time at all -- in which case the conservative
    14:00 default applies. Returning ``None`` rather than a guess is the point:
    an unverified time must err late.

    Note we ignore the stated EST/EDT suffix and localise to
    ``America/New_York``, which resolves the correct offset for the date
    automatically. Trusting the suffix would mean trusting the Fed's proofreader
    on daylight-saving boundaries.
    """
    match = _RELEASE_TIME_RE.search(text[:1500])
    if match is None:
        return None
    hour, minute, meridiem = int(match.group(1)), int(match.group(2)), match.group(3).lower()
    if not (1 <= hour <= 12) or not (0 <= minute <= 59):
        return None
    if meridiem == "p" and hour != 12:
        hour += 12
    elif meridiem == "a" and hour == 12:
        hour = 0
    return time(hour, minute)


def resolve_release_datetime(
    doc_date: date, cfg: Config, text: str = ""
) -> tuple[datetime, str, time | None]:
    """Combine a document date with its release time, timezone-aware.

    Returns ``(release_datetime, release_time_source, parsed_time)``.

    Precedence, most to least authoritative:

    1. ``manual_override`` -- an explicit, researched human decision. Wins so
       that a regex change can never silently overturn a verified value.
    2. ``parsed_from_document`` -- the time the artifact states about itself.
       Evidence, not assumption.
    3. ``scheduled_default`` (14:00 ET) -- applies when nothing is known. The
       config schema *enforces* that this is after the session open, so an
       unverified document can never win same-session entry.

    ``parsed_time`` is returned alongside even when an override wins, so the
    caller can report disagreements. An override that contradicts the document
    is exactly the kind of thing that should be surfaced rather than buried.
    """
    parsed = extract_release_time(text) if text else None

    override = cfg.scrape.release_time_overrides.get(doc_date)
    if override is not None:
        return datetime.combine(doc_date, override, tzinfo=ET), "manual_override", parsed

    if parsed is not None:
        return datetime.combine(doc_date, parsed, tzinfo=ET), "parsed_from_document", parsed

    return (
        datetime.combine(doc_date, cfg.scrape.scheduled_release_time_et, tzinfo=ET),
        "scheduled_default",
        None,
    )


def build_documents_table(cfg: Config, docs: pd.DataFrame) -> pd.DataFrame:
    """Parse every downloaded document into ``data/interim/documents.parquet``.

    Adds ``text``, ``n_chars``, ``selector_used``, ``release_datetime``,
    ``release_time_source``, and ``is_flagged`` (text shorter than
    ``min_text_chars``). Flagged rows are **kept**, not dropped, so the phase
    report can prove ``n_scraped == n_parsed + n_flagged``.
    """
    repo_root = cfg.paths.resolve("data_raw").parent.parent
    records: list[dict] = []

    for _, row in docs.iterrows():
        path = repo_root / row["raw_path"]
        if not path.exists():
            # Non-200 responses are never written to disk, so a missing file
            # here means the fetch failed. Recorded and flagged, never dropped.
            release_dt, source, _ = resolve_release_datetime(row["doc_date"], cfg)
            records.append(
                {
                    **row.to_dict(),
                    "text": "",
                    "n_chars": 0,
                    "selector_used": "MISSING_FILE",
                    "release_datetime": release_dt,
                    "release_time_source": source,
                    "parsed_release_time": None,
                    "is_error_page": False,
                    "is_flagged": True,
                }
            )
            continue

        text, selector = parse_statement(path.read_bytes())
        release_dt, source, parsed_time = resolve_release_datetime(row["doc_date"], cfg, text)

        is_error_page = looks_like_error_page(text)
        too_short = len(text) < cfg.scrape.min_text_chars

        records.append(
            {
                **row.to_dict(),
                "text": text,
                "n_chars": len(text),
                "selector_used": selector,
                "release_datetime": release_dt,
                "release_time_source": source,
                "parsed_release_time": parsed_time.isoformat() if parsed_time else None,
                "is_error_page": is_error_page,
                # Two independent reasons to distrust a parse. Length alone is
                # NOT enough: a Fed 404 body is ~1,170 characters of clean prose.
                "is_flagged": too_short or is_error_page,
            }
        )

    df = pd.DataFrame(records).sort_values(["doc_type", "doc_date"]).reset_index(drop=True)
    out = cfg.paths.resolve("data_interim") / "documents.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


def domain_of(url: str) -> str:
    return urlparse(url).netloc
