"""Phase 1 -- the leak guard. Maps ``release_datetime`` to a tradable entry.

This module is where look-ahead bias lives or dies.

**Look-ahead bias** (PLAN.md section 2.1): using, in a decision made at time
*t*, information that was not public until some later time *t+k*. It is the most
common reason a backtest looks brilliant and live trading loses money.

The alignment rule (PLAN.md section 3.3)
----------------------------------------
    A signal computed from a document is effective at the **open of the first
    trading day strictly after** its ``release_datetime``.

Implementation is a single ``searchsorted``, which is a binary search over a
sorted array returning an insertion index. Given the sorted array of SPY
trading days::

    idx = trading_days.searchsorted(release_datetime, side="right")
    entry_date = trading_days[idx]

``side="right"`` gives the first element *strictly greater than* the key, which
is the rule verbatim -- no loops, no off-by-one to argue about.

Why not ``release_date + 1 day``?
---------------------------------
Because it is wrong on exactly the cases that matter most. The 2020-03-15
emergency cut to 0-0.25% was announced on a **Sunday evening**; date arithmetic
would target a non-trading Monday-that-isn't (it happens to work here, but only
by luck). The 2008-10-08 coordinated cut was released at **07:00 ET**, before
the open, which a date-only rule cannot distinguish from a 14:00 release at all.
Timestamps must be tz-aware ``America/New_York``, and the next-day lookup must
go through the real trading calendar.

Forward returns
---------------
For horizon *h* trading days::

    R = (P_exit / P_entry) - 1

with ``P_entry`` the adjusted Open on ``entry_date`` and ``P_exit`` the adjusted
Open *h* trading days later. Note both legs use the same price field, so the
return is a clean, executable open-to-open return.

**Overlap:** FOMC events are ~32 trading days apart (8/year), so consecutive
windows do not overlap even at *h*=20. Overlapping forward returns are
autocorrelated and inflate apparent significance; :func:`check_overlap` reports
the minimum spacing so this stays a verified fact rather than an assumption.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config


def next_trading_open(release_datetimes: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Map each release timestamp to the first trading day strictly after it.

    Parameters
    ----------
    release_datetimes
        tz-aware (America/New_York) Series of publication timestamps.
    trading_days
        Sorted DatetimeIndex of valid sessions (from the SPY price index).

    Returns
    -------
    Series of entry dates, ``NaT`` where the release post-dates the price data
    (i.e. the most recent meeting, before its next session has occurred).
    """
    raise NotImplementedError("Phase 1")


def forward_returns(
    entry_dates: pd.Series,
    prices: pd.DataFrame,
    horizons: list[int],
    entry_field: str = "Open",
    exit_field: str = "Open",
) -> pd.DataFrame:
    """Compute ``fwd_ret_{h}`` for each horizon in ``horizons``.

    NaN where the exit date falls beyond the end of the price series -- an
    unavoidable tail of at most ``max(horizons)`` sessions, never a silent zero.
    """
    raise NotImplementedError("Phase 1")


def check_overlap(entry_dates: pd.Series, trading_days: pd.DatetimeIndex) -> dict:
    """Report the spacing between consecutive events, in trading days.

    Returns min/median/max spacing plus the largest horizon that remains
    strictly non-overlapping. Used to justify (rather than assume) that the
    forward-return series is not autocorrelated by construction.
    """
    raise NotImplementedError("Phase 1")


def build_panel(cfg: Config) -> pd.DataFrame:
    """Produce ``data/processed/panel.parquet`` -- the Phase 1 deliverable.

    Columns: ``event_date``, ``release_datetime``, ``release_time_source``,
    ``entry_date``, ``doc_type``, ``raw_path``, and one ``fwd_ret_{h}`` per
    configured horizon.

    Invariant asserted here and re-asserted in
    ``tests/test_align_no_lookahead.py``: ``entry_date > release_datetime`` for
    **every** row, and every ``entry_date`` is a member of ``trading_days``.
    """
    raise NotImplementedError("Phase 1")
