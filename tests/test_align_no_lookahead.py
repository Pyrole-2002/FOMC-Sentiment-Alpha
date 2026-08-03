"""Phase 1 leak canary: no signal may be actionable before its text exists.

Two layers, deliberately:

**Synthetic tests** exercise the alignment logic against a hand-built calendar.
They are fast, deterministic, need no network, and can construct the awkward
cases (Sunday releases, pre-open releases, holidays) on demand rather than
hoping the real data contains them.

**Integration tests** assert the same invariants on the actual built panel.
They catch the failure a synthetic test cannot: that the real data does not look
like the data you imagined.

The invariants (PLAN.md sections 3.3 and 7):

1. The entry *open instant* is strictly after ``release_datetime``. Equality is
   a failure -- it would mean transacting at the very moment of publication.
2. Every ``entry_date`` is a real trading session.
3. ``entry_date`` is the **first** qualifying session, not merely some later
   one. Skipping ahead would be conservative rather than leaky, but it would
   silently change the horizon being measured.
4. The verified emergency meetings align as documented.
5. Forward-return NaNs appear only in the unavoidable tail.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.align import (
    check_overlap,
    forward_returns,
    next_trading_open,
    session_open_instants,
)
from src.config import load_config

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Synthetic fixtures -- a tiny hand-built market
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def sessions() -> pd.DatetimeIndex:
    """Mar 2020 weekdays, minus a fabricated holiday on Mar 12.

    Chosen because it contains the real Sunday release (Mar 15) and the real
    8-session gap between the Mar 3 and Mar 15 statements.
    """
    days = pd.bdate_range("2020-03-02", "2020-03-31")
    return days[days != pd.Timestamp("2020-03-12")]


def _et(y: int, m: int, d: int, hh: int, mm: int = 0) -> pd.Timestamp:
    return pd.Timestamp(datetime(y, m, d, hh, mm, tzinfo=ET))


# ---------------------------------------------------------------------------
# Synthetic: the core rule
# ---------------------------------------------------------------------------


def test_afternoon_release_enters_next_session(cfg, sessions):
    """A 14:00 release has missed that day's 09:30 open -> next session."""
    releases = pd.Series([_et(2020, 3, 4, 14, 0)])
    entries = next_trading_open(releases, sessions, cfg)
    assert entries.iloc[0] == pd.Timestamp("2020-03-05")


def test_pre_open_release_enters_the_same_session(cfg, sessions):
    """An 08:00 release has NOT missed that day's 09:30 open -> same session.

    This is the case a day-granularity rule gets wrong. 2008-10-08 (07:00 ET)
    and 2020-03-23 (08:00 ET) are both pre-open, and both are among the
    highest-impact events in the sample, so entering them a session late
    concentrates the error exactly where it costs most.
    """
    releases = pd.Series([_et(2020, 3, 4, 8, 0)])
    entries = next_trading_open(releases, sessions, cfg)
    assert entries.iloc[0] == pd.Timestamp("2020-03-04")


def test_release_exactly_at_the_open_is_not_tradable_that_session(cfg, sessions):
    """Strictly-after: a release at exactly 09:30 does NOT get that open.

    Equality must fail. Being able to transact at the precise instant of
    publication is not a real capability, and admitting it would be the thin end
    of the look-ahead wedge.
    """
    releases = pd.Series([_et(2020, 3, 4, 9, 30)])
    entries = next_trading_open(releases, sessions, cfg)
    assert entries.iloc[0] == pd.Timestamp("2020-03-05")


def test_sunday_evening_release_enters_monday(cfg, sessions):
    """2020-03-15 17:00 ET was a Sunday -- the COVID cut to 0-0.25%."""
    releases = pd.Series([_et(2020, 3, 15, 17, 0)])
    entries = next_trading_open(releases, sessions, cfg)
    assert entries.iloc[0] == pd.Timestamp("2020-03-16")


def test_release_skips_a_holiday(cfg, sessions):
    """Mar 12 is not a session here, so a Mar 11 afternoon release enters Mar 13."""
    releases = pd.Series([_et(2020, 3, 11, 14, 0)])
    entries = next_trading_open(releases, sessions, cfg)
    assert entries.iloc[0] == pd.Timestamp("2020-03-13")


def test_entry_is_the_first_qualifying_session(cfg, sessions):
    """Never skip ahead: the entry is the earliest session open after release."""
    releases = pd.Series([_et(2020, 3, 4, 14, 0), _et(2020, 3, 10, 8, 0)])
    entries = next_trading_open(releases, sessions, cfg)
    opens = session_open_instants(sessions, cfg)

    for release, entry in zip(releases, entries, strict=True):
        candidates = sessions[opens > release]
        assert entry == candidates[0], "entry must be the FIRST session open after release"


def test_release_after_all_data_yields_nat(cfg, sessions):
    releases = pd.Series([_et(2020, 4, 15, 14, 0)])
    entries = next_trading_open(releases, sessions, cfg)
    assert pd.isna(entries.iloc[0])


def test_naive_timestamps_are_rejected(cfg, sessions):
    """Timezone-naive input must raise, not be silently reinterpreted.

    A naive 14:00 read as UTC is 09:00 ET -- before the open -- which would flip
    an ordinary statement into same-session entry. That is a leak introduced by
    a missing timezone, so it must be impossible rather than merely unlikely.
    """
    releases = pd.Series([pd.Timestamp("2020-03-04 14:00")])
    with pytest.raises(ValueError, match="timezone-aware"):
        next_trading_open(releases, sessions, cfg)


def test_forward_returns_count_sessions_not_calendar_days(cfg, sessions):
    """h is a session count, so weekends and the holiday are stepped over."""
    prices = pd.DataFrame(
        {"Open": range(100, 100 + len(sessions)), "Close": range(100, 100 + len(sessions))},
        index=sessions,
    ).astype(float)

    entries = pd.Series([pd.Timestamp("2020-03-04")])
    out = forward_returns(entries, prices, [1, 5])
    # Open rises by exactly 1.0 per session, starting at 100 on 2020-03-02.
    # 2020-03-04 is the 3rd session -> Open 102.
    assert out["fwd_ret_1"].iloc[0] == pytest.approx(103 / 102 - 1)
    assert out["fwd_ret_5"].iloc[0] == pytest.approx(107 / 102 - 1)


def test_forward_returns_nan_past_the_end(cfg, sessions):
    prices = pd.DataFrame({"Open": 1.0, "Close": 1.0}, index=sessions)
    entries = pd.Series([sessions[-2]])
    out = forward_returns(entries, prices, [1, 20])
    assert pd.notna(out["fwd_ret_1"].iloc[0])
    assert pd.isna(out["fwd_ret_20"].iloc[0])


def test_check_overlap_flags_close_events(sessions):
    """Overlapping forward returns are autocorrelated and inflate significance."""
    entries = pd.Series([pd.Timestamp("2020-03-03"), pd.Timestamp("2020-03-16")])
    report = check_overlap(entries, sessions, [1, 5, 20])
    assert report["n_events"] == 2
    assert report["overlapping_pairs"][1] == []
    assert len(report["overlapping_pairs"][20]) == 1, "h=20 must flag this pair"


# ---------------------------------------------------------------------------
# Integration -- the real panel
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel(cfg):
    path = cfg.paths.resolve("data_processed") / "panel.parquet"
    if not path.exists():
        pytest.skip("panel.parquet not built; run scripts/build_panel.py")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def real_sessions(cfg):
    from src.prices import download_prices, trading_days

    path = cfg.paths.resolve("data_processed") / f"{cfg.data.ticker.lower()}_prices.parquet"
    if not path.exists():
        pytest.skip("prices not downloaded; run scripts/fetch_prices.py")
    return trading_days(download_prices(cfg))


def test_real_entry_strictly_after_release(panel, cfg):
    have = panel[panel["entry_date"].notna()]
    opens = session_open_instants(pd.DatetimeIndex(have["entry_date"]), cfg)
    releases = pd.to_datetime(have["release_datetime"]).dt.tz_convert(ET)
    assert (opens > releases.to_numpy()).all(), "entry open must post-date the release"


def test_real_entry_is_a_trading_session(panel, real_sessions):
    have = panel[panel["entry_date"].notna()]
    assert pd.DatetimeIndex(have["entry_date"]).isin(real_sessions).all()


def test_real_entry_is_the_first_qualifying_session(panel, real_sessions, cfg):
    have = panel[panel["entry_date"].notna()]
    opens = session_open_instants(real_sessions, cfg)
    releases = pd.to_datetime(have["release_datetime"]).dt.tz_convert(ET)

    for release, entry in zip(releases, have["entry_date"], strict=True):
        expected = real_sessions[opens > release][0]
        assert entry == expected, f"release {release} should enter {expected}, got {entry}"


@pytest.mark.parametrize(
    "doc_date,expected_entry,why",
    [
        (date(2020, 3, 15), date(2020, 3, 16), "Sunday 17:00 ET -> Monday open"),
        (date(2008, 10, 8), date(2008, 10, 8), "07:00 ET pre-open -> SAME session"),
        (date(2001, 9, 17), date(2001, 9, 18), "14:00 default -> next session"),
    ],
)
def test_known_emergency_meetings_align_correctly(panel, doc_date, expected_entry, why):
    """The cases where naive date arithmetic goes wrong.

    ⚠️ Compare via ``pd.Timestamp``, not a raw ``datetime.date``. When
    ``panel.doc_date`` was normalised to datetime64, ``== date(...)`` silently
    matched nothing, so these three checks began SKIPPING rather than failing --
    and a skip reads as green in the summary line. A conditional skip that
    depends on a dtype is really an assertion in disguise, so the presence of
    each date is now asserted outright.
    """
    matches = panel[panel["doc_date"] == pd.Timestamp(doc_date)]
    assert not matches.empty, (
        f"{doc_date} must be in the sample -- if this date was legitimately "
        "excluded, update the parametrisation deliberately rather than skipping."
    )
    actual = pd.Timestamp(matches["entry_date"].iloc[0]).date()
    assert actual == expected_entry, why


def test_real_forward_returns_nan_only_in_the_tail(panel, cfg):
    """NaNs are permitted only for events whose exit exceeds the price data."""
    max_h = max(cfg.backtest.horizons_days)
    col = f"fwd_ret_{max_h}"
    n_nan = int(panel[col].isna().sum())
    assert n_nan <= 2, f"{n_nan} NaNs at h={max_h}; expected only the unavoidable tail"

    # Shorter horizons can never have MORE missing values than longer ones.
    prev = None
    for h in sorted(cfg.backtest.horizons_days):
        n = int(panel[f"fwd_ret_{h}"].isna().sum())
        if prev is not None:
            assert n >= prev, "NaN count must be non-decreasing in the horizon"
        prev = n


def test_no_duplicate_events(panel):
    assert not panel["doc_date"].duplicated().any()


def test_doc_date_is_datetime64(panel):
    """Pin the join-key dtype across every artifact.

    Parquet round-trips a python ``date`` column as object dtype while the
    sentiment cache yields datetime64, and merging the two raises. One canonical
    dtype, asserted, beats a coercion at each call site.
    """
    assert pd.api.types.is_datetime64_any_dtype(panel["doc_date"])


def test_release_times_are_conservative_by_default(panel, cfg):
    """Every unverified release must use the late (post-open) default.

    The safety principle in test form: a document whose time we did not verify
    must never be stamped with a pre-open time, because that would grant it
    same-session entry on evidence we do not have.
    """
    defaults = panel[panel["release_time_source"] == "scheduled_default"]
    times = pd.to_datetime(defaults["release_datetime"]).dt.tz_convert(ET).dt.time
    assert (times > time(9, 30)).all() if len(times) else True
