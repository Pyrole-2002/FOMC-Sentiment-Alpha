"""Phase 1 -- the leak guard. Maps ``release_datetime`` to a tradable entry.

This module is where look-ahead bias lives or dies.

**Look-ahead bias** (PLAN.md section 2.1): using, in a decision made at time
*t*, information that was not public until some later time *t+k*. It is the most
common reason a backtest looks brilliant and live trading loses money.

The alignment rule
------------------
    A signal computed from a document is effective at the **first session open
    that occurs strictly after** its ``release_datetime``.

🔧 Note the refinement over PLAN.md section 3.3, forced by evidence found in
Phase 1. The plan said "the open of the first trading *day* strictly after the
release", which silently assumes every release lands during or after a session.
Several do not::

    2008-10-08  07:00 ET   coordinated emergency cut, released PRE-open
    2020-03-23  08:00 ET   pre-open
    2020-03-15  17:00 ET   a SUNDAY evening

A day-granularity rule enters the 2008-10-08 statement on **2008-10-09**, a full
session late -- and these are the highest-impact events in the sample, so the
error is concentrated exactly where it costs most.

Comparing against the **session open instant** (09:30 ET) instead fixes this
with no special-casing at all:

* a 14:00 release has missed that day's 09:30 open  -> entry is the NEXT open
* an 08:00 release has not                          -> entry is THAT day's open

Same rule, correctly applied. The 14:00 default therefore still produces
next-day entry for every ordinary statement, exactly as the plan intended.

⚠️ **The safety principle.** The 14:00 default governs every document whose true
release time we have not verified, and the config schema *enforces* that this
default is after the open. Erring late costs signal; erring early would let us
trade at an open that preceded the text. Never guess early.

Forward returns
---------------
For horizon *h* trading days::

    R = (P_exit / P_entry) - 1

with ``P_entry`` the adjusted Open on ``entry_date`` and ``P_exit`` the adjusted
Open *h* sessions later. Both legs use the same price field on the same
adjustment basis, so the return is clean and executable (PLAN.md 4.3).
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.config import Config
from src.prices import download_prices, trading_days

ET = ZoneInfo("America/New_York")


def session_open_instants(sessions: pd.DatetimeIndex, cfg: Config) -> pd.DatetimeIndex:
    """Turn midnight-stamped session dates into tz-aware opening instants.

    The price index labels each session by date at midnight. A date is not an
    instant, so it cannot be compared with a release timestamp without first
    deciding *which* instant in that session we mean. We mean the opening
    auction -- the first moment we could transact.
    """
    return pd.DatetimeIndex(
        sessions.normalize()
        + pd.Timedelta(hours=cfg.scrape.session_open_time_et.hour)
        + pd.Timedelta(minutes=cfg.scrape.session_open_time_et.minute)
    ).tz_localize(ET)


def next_trading_open(
    release_datetimes: pd.Series, sessions: pd.DatetimeIndex, cfg: Config
) -> pd.Series:
    """Map each release instant to the first session open strictly after it.

    Implemented as a single ``searchsorted``, a binary search over a sorted
    array returning an insertion index. With ``side="right"`` the returned index
    points at the first element **strictly greater than** the key -- which is
    the alignment rule verbatim. No loop, no ``+ 1``, no weekend branch, and no
    off-by-one to argue about.

    Parameters
    ----------
    release_datetimes
        tz-aware Series of publication instants.
    sessions
        Sorted DatetimeIndex of trading days (from the SPY price index).

    Returns
    -------
    Series of entry dates (tz-naive, matching the price index), ``NaT`` where
    the release post-dates the price data -- i.e. the most recent statement,
    before its next session has happened.
    """
    opens = session_open_instants(sessions, cfg)

    rel = pd.to_datetime(release_datetimes)
    if rel.dt.tz is None:
        raise ValueError(
            "release_datetimes must be timezone-aware. Naive timestamps silently "
            "shift by hours the moment anything touches UTC, which moves entry "
            "dates across midnight and corrupts the leak guard."
        )
    rel = rel.dt.tz_convert(ET)

    idx = opens.searchsorted(rel.to_numpy(), side="right")

    out = pd.Series(pd.NaT, index=release_datetimes.index, dtype="datetime64[ns]")
    valid = idx < len(sessions)
    out.loc[valid] = sessions[idx[valid]]
    return out


def forward_returns(
    entry_dates: pd.Series,
    prices: pd.DataFrame,
    horizons: list[int],
    entry_field: str = "Open",
    exit_field: str = "Open",
) -> pd.DataFrame:
    """Compute ``fwd_ret_{h}`` for each horizon, in trading days.

    *h* counts **sessions, not calendar days** -- we index into the price array
    rather than adding timedeltas, so weekends and holidays are stepped over
    automatically.

    NaN where the exit session falls beyond the end of the price series: an
    unavoidable tail of at most ``max(horizons)`` events, never a silent zero.
    """
    sessions = trading_days(prices)
    entry_pos = pd.Series(
        sessions.get_indexer(pd.DatetimeIndex(entry_dates)), index=entry_dates.index
    )

    entry_px = prices[entry_field].to_numpy()
    exit_px = prices[exit_field].to_numpy()
    n = len(sessions)
    pos = entry_pos.to_numpy()

    out = pd.DataFrame(index=entry_dates.index)
    for h in horizons:
        values = np.full(len(pos), np.nan)
        # `get_indexer` returns -1 for a date absent from the index (an
        # unresolved entry), so the lower bound filters those out and the upper
        # bound filters exits that run past the end of the price series. Both
        # stay NaN rather than silently becoming a number.
        usable = (pos >= 0) & (pos + h < n)
        values[usable] = exit_px[pos[usable] + h] / entry_px[pos[usable]] - 1.0
        out[f"fwd_ret_{h}"] = values
    return out


def check_overlap(entry_dates: pd.Series, sessions: pd.DatetimeIndex, horizons: list[int]) -> dict:
    """Report the spacing between consecutive events, in trading days.

    **Why this matters.** The IC t-statistic in PLAN.md section 14.4 assumes the
    observations are independent. If two events are closer together than the
    horizon, their forward-return windows *overlap*: they share price moves, so
    the returns are autocorrelated and the effective sample size is smaller than
    ``n``. That inflates apparent significance -- a real way to fool yourself.

    ⚠️ **Measured on the real 225-event sample, the tidy story is false.** The
    median gap is 30 sessions, but the *minimum* is **3**, and overlap is
    substantial at long horizons::

        h= 1:  0 pairs      h=10:  8 pairs
        h= 3:  0 pairs      h=20: 16 pairs  (7% of the sample)
        h= 5:  1 pair

    Reasoning from the average (~32 sessions) and ignoring the minimum is what
    made PLAN.md section 4.3 originally claim overlap was negligible. Unscheduled
    meetings cluster *inside* the gaps between scheduled ones -- and they cluster
    during crises (2007-08, 2008-01, 2008-03, 2008-10, 2020-03), so the overlap
    concentrates in the highest-variance episodes, the worst possible place.

    This function therefore *enumerates* every offending pair rather than
    letting the assumption stand unexamined.
    """
    ordered = pd.DatetimeIndex(entry_dates.dropna().sort_values().unique())
    positions = sessions.get_indexer(ordered)
    positions = positions[positions >= 0]
    gaps = np.diff(positions)

    offenders = {}
    for h in horizons:
        bad = [
            (str(ordered[i].date()), str(ordered[i + 1].date()), int(gaps[i]))
            for i in range(len(gaps))
            if gaps[i] < h
        ]
        offenders[h] = bad

    return {
        "n_events": len(ordered),
        "min_gap_sessions": int(gaps.min()) if len(gaps) else None,
        "median_gap_sessions": float(np.median(gaps)) if len(gaps) else None,
        "max_gap_sessions": int(gaps.max()) if len(gaps) else None,
        "max_non_overlapping_horizon": int(gaps.min()) if len(gaps) else None,
        "overlapping_pairs": offenders,
    }


def build_panel(cfg: Config, documents: pd.DataFrame | None = None) -> pd.DataFrame:
    """Produce ``data/processed/panel.parquet`` -- the Phase 1 deliverable.

    Applies, in order: statements only -> drop flagged parses -> sample-start
    filter (the disclosure-regime cut) -> optional unscheduled filter. Each step
    is counted so the phase report can prove nothing vanished silently.

    Invariants asserted here and re-asserted in
    ``tests/test_align_no_lookahead.py``: ``entry_open_instant >
    release_datetime`` for **every** row, and every ``entry_date`` is a session.
    """
    if documents is None:
        documents = pd.read_parquet(cfg.paths.resolve("data_interim") / "documents.parquet")

    prices = download_prices(cfg)
    sessions = trading_days(prices)

    counts = {"discovered": len(documents)}

    df = documents[documents["doc_type"] == "statement"].copy()
    counts["statements"] = len(df)

    df = df[~df["is_flagged"]].copy()
    counts["after_dropping_flagged"] = len(df)

    df = df[df["doc_date"] >= cfg.data.start_date].copy()
    counts["after_sample_start"] = len(df)

    if not cfg.sample.include_unscheduled:
        df = df[df["is_scheduled"] == True].copy()  # noqa: E712
    counts["after_unscheduled_filter"] = len(df)

    df = df.sort_values("doc_date").reset_index(drop=True)

    df["entry_date"] = next_trading_open(df["release_datetime"], sessions, cfg)
    counts["with_entry_date"] = int(df["entry_date"].notna().sum())

    returns = forward_returns(
        df["entry_date"],
        prices,
        cfg.backtest.horizons_days,
        entry_field=cfg.data.entry_price_field,
        exit_field=cfg.data.exit_price_field,
    )
    panel = pd.concat([df.drop(columns=["text"], errors="ignore"), returns], axis=1)

    _assert_no_lookahead(panel, sessions, cfg)

    out = cfg.paths.resolve("data_processed") / "panel.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    panel.attrs["counts"] = counts
    return panel


def _assert_no_lookahead(panel: pd.DataFrame, sessions: pd.DatetimeIndex, cfg: Config) -> None:
    """Fail loudly, at build time, if the leak guard is violated.

    Duplicated in the test suite on purpose: a test protects you when you run
    it, an assertion protects you every time the pipeline runs. For the one
    error that invalidates the entire project, both is correct.
    """
    have_entry = panel[panel["entry_date"].notna()]
    if have_entry.empty:
        return

    entry_opens = session_open_instants(pd.DatetimeIndex(have_entry["entry_date"]), cfg)
    releases = pd.to_datetime(have_entry["release_datetime"]).dt.tz_convert(ET)

    violations = entry_opens <= releases.to_numpy()
    if violations.any():
        bad = have_entry.loc[violations, ["doc_date", "release_datetime", "entry_date"]]
        raise AssertionError(
            f"LOOK-AHEAD DETECTED in {int(violations.sum())} rows: the entry open is "
            f"at or before the release instant.\n{bad.to_string()}"
        )

    not_a_session = ~pd.DatetimeIndex(have_entry["entry_date"]).isin(sessions)
    if not_a_session.any():
        raise AssertionError(f"{int(not_a_session.sum())} entry dates are not trading sessions.")
