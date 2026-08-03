"""Phase 1 -- SPY daily price loader and the authoritative trading calendar.

**SPY** is the SPDR S&P 500 ETF Trust, the most liquid proxy for US large-cap
equities. It began trading 1993-01-22, which pre-dates the first regular FOMC
statement (February 1994), so the price series never limits the sample.

Price adjustment (PLAN.md section 3.4)
--------------------------------------
**Adjustment** means rewriting historical prices so dividends and splits do not
appear as price gaps. SPY distributes ~1.3%/yr across four dividends; unadjusted
prices show each as a phantom one-day loss.

We use ``auto_adjust=True``, which back-adjusts **all four** OHLC columns on a
single consistent basis, so ``Open`` *is* the adjusted open. The alternative
(``auto_adjust=False``) returns raw OHLC plus one extra ``Adj Close`` column and
**no** ``Adj Open``, so an open-to-open return would mix two price bases.

Trading calendar
----------------
The **SPY price index is the source of truth** for "is this a trading day."
This is self-consistent: a day we cannot price is a day we cannot trade.
``pandas_market_calendars`` is used only to *validate* that index against the
official NYSE calendar -- never to fill it in. Pick one source of truth, then
use the second to audit it.

Provenance
----------
``yfinance`` scrapes an undocumented endpoint and can rate-limit, change shape,
or return partial history without raising. The parquet cache is therefore the
working copy, and every download writes a sidecar JSON recording what was
fetched and when -- the price-data analogue of ``data/raw/manifest.csv``.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from src.config import Config

# Columns we keep. yfinance may also return "Dividends"/"Stock Splits" when
# actions=True; we do not request them because auto_adjust has already folded
# their effect into the prices, and keeping them would invite double-counting.
OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# SPY's first trading day on the exchange.
SPY_INCEPTION = date(1993, 1, 22)

# Yahoo's SPY history actually begins 1993-01-29 -- five sessions after listing.
# This is a known, stable quirk of the provider, not a truncation bug, and it is
# harmless here because the sample starts in 1994. Recorded so that a future
# reader sees a documented fact rather than an unexplained discrepancy.
YAHOO_SPY_FIRST_SESSION = date(1993, 1, 29)


def _cache_paths(cfg: Config) -> tuple[Path, Path]:
    """Return (parquet path, provenance-sidecar path)."""
    processed = cfg.paths.resolve("data_processed")
    ticker = cfg.data.ticker.lower()
    return processed / f"{ticker}_prices.parquet", processed / f"{ticker}_prices_meta.json"


def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Collapse yfinance's MultiIndex columns to plain OHLCV names.

    yfinance returns a MultiIndex (field, ticker) whenever it thinks more than
    one ticker might be involved, and its heuristics for that have changed
    across versions. Rather than depend on which shape today's version returns,
    we normalise both shapes here. Guessing wrong would silently produce a
    single column of tuples that still *looks* like a DataFrame.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # Drop whichever level holds the ticker symbol, keeping the field names.
    for level in range(df.columns.nlevels):
        values = set(df.columns.get_level_values(level))
        if values <= {ticker, ticker.upper(), ticker.lower()}:
            return df.droplevel(level, axis=1)

    # Fallback: assume (field, ticker) ordering, the historical default.
    return df.droplevel(-1, axis=1)


def download_prices(cfg: Config, force_refresh: bool = False) -> pd.DataFrame:
    """Download adjusted daily OHLCV and cache to parquet.

    Returns a DataFrame indexed by **timezone-naive** ``DatetimeIndex`` of
    trading days, with columns ``Open/High/Low/Close/Volume``, all split- and
    dividend-adjusted.

    Why tz-naive here, when ``release_datetime`` is tz-aware? Because a daily
    *bar* is a label for a session, not an instant. Attaching a timezone to it
    invites the illusion that "2020-03-16 00:00 ET" is meaningful, when the
    session actually spans 09:30-16:00. The comparison that matters -- release
    instant versus session -- is handled explicitly in ``align.py``, which
    localises the session boundary rather than pretending the bar has one.

    Parameters
    ----------
    force_refresh
        Re-download even if the cache exists. The default is to reuse the cache,
        because the network is unreliable and the historical data is immutable.
    """
    import yfinance as yf

    parquet_path, meta_path = _cache_paths(cfg)

    if parquet_path.exists() and not force_refresh:
        return pd.read_parquet(parquet_path)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # `period="max"` rather than an explicit start: we want the complete trading
    # calendar, including sessions before the first FOMC statement, so that
    # `searchsorted` in align.py can never fall off the front of the array.
    end = cfg.data.end_date.isoformat() if cfg.data.end_date else None
    raw = yf.download(
        cfg.data.ticker,
        period="max" if end is None else None,
        start=None if end is None else SPY_INCEPTION.isoformat(),
        end=end,
        interval="1d",
        auto_adjust=cfg.data.auto_adjust,
        actions=False,
        progress=False,
        threads=False,
    )

    if raw is None or raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for {cfg.data.ticker!r}. This is usually "
            "a transient rate limit or a network failure, not a code bug -- retry "
            "before investigating."
        )

    df = _flatten_columns(raw, cfg.data.ticker)

    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"yfinance response is missing {missing}; got {list(df.columns)}. "
            "The provider changed its response shape -- inspect before trusting."
        )
    df = df[OHLCV].copy()

    # Normalise the index: drop any timezone, floor to midnight, sort, dedupe.
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df.columns.name = None  # yfinance labels the column axis "Price"; cosmetic noise
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # A row with no close is not a session we could have traded. Drop it loudly
    # rather than carrying a NaN into a return calculation.
    n_before = len(df)
    df = df.dropna(subset=["Open", "Close"])
    n_dropped = n_before - len(df)

    _write_meta(
        meta_path,
        cfg=cfg,
        yf_version=yf.__version__,
        n_rows=len(df),
        n_dropped=n_dropped,
        first=df.index[0].date(),
        last=df.index[-1].date(),
    )
    df.to_parquet(parquet_path)
    return df


def _write_meta(meta_path: Path, *, cfg: Config, yf_version: str, **fields) -> None:
    """Record what was downloaded and when.

    The provenance analogue of ``data/raw/manifest.csv``: `period="max"` means
    the result depends on the date it was fetched, so that date must be recorded
    or the "reproducible" claim is false.
    """
    payload = {
        "ticker": cfg.data.ticker,
        "auto_adjust": cfg.data.auto_adjust,
        "yfinance_version": yf_version,
        "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **{k: str(v) for k, v in fields.items()},
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def trading_days(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the sorted index of valid trading sessions.

    This is **the** trading calendar for the project. ``align.py`` does a
    ``searchsorted`` into exactly this array, so anything absent here is, by
    definition, a day on which we did not trade.
    """
    idx = pd.DatetimeIndex(prices.index)
    if not idx.is_monotonic_increasing:
        raise ValueError("price index is not sorted; align.searchsorted would be wrong")
    if idx.has_duplicates:
        raise ValueError("price index has duplicate dates")
    return idx


def validate_against_nyse(prices: pd.DataFrame) -> pd.DataFrame:
    """Cross-check the price index against the official NYSE calendar.

    Returns a frame of discrepancies with columns ``date`` and ``issue``:

    ``missing_from_prices``
        NYSE says a session existed; we have no bar. A **download gap**. This is
        the dangerous direction: ``searchsorted`` would skip to the following
        session, silently shifting an entry date by one and corrupting every
        forward return computed from it.
    ``not_an_nyse_session``
        We have a bar on a date NYSE says was closed. A provider data error.

    An empty frame is the pass condition. Note this *reports*; it never repairs
    -- silently filling a gap would fabricate a price we never observed.
    """
    import pandas_market_calendars as mcal

    idx = trading_days(prices)
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=idx[0], end_date=idx[-1])
    nyse_days = pd.DatetimeIndex(schedule.index).normalize()

    ours = set(idx)
    theirs = set(nyse_days)

    rows = [{"date": d, "issue": "missing_from_prices"} for d in sorted(theirs - ours)]
    rows += [{"date": d, "issue": "not_an_nyse_session"} for d in sorted(ours - theirs)]

    return pd.DataFrame(rows, columns=["date", "issue"])


def summarise(prices: pd.DataFrame) -> dict:
    """Human-readable facts about the loaded series, for the phase report."""
    idx = trading_days(prices)
    return {
        "n_sessions": len(idx),
        "first_session": idx[0].date().isoformat(),
        "last_session": idx[-1].date().isoformat(),
        "years_covered": round((idx[-1] - idx[0]).days / 365.25, 1),
        "sessions_per_year": round(len(idx) / ((idx[-1] - idx[0]).days / 365.25), 1),
        # Yahoo's history starts 1993-01-29, five sessions after SPY listed on
        # 1993-01-22. Expected and harmless (the sample starts 1994); flagged
        # only if the start drifts LATER than that known floor.
        "history_starts_as_expected": idx[0].date() <= YAHOO_SPY_FIRST_SESSION,
        "any_nan": bool(prices[OHLCV].isna().any().any()),
    }
