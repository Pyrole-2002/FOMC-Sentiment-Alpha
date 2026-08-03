"""Phase 1 leak canary: no signal may be actionable before its text exists.

Currently skipped -- ``build_panel`` lands in Phase 1. The invariants are
written down now so the implementation is developed *against* them rather than
having them retrofitted to whatever it happens to produce.

The invariants (PLAN.md sections 3.3 and 7):

1. ``entry_date > release_datetime`` for **every** row, strictly. Equality is a
   failure: it would mean trading at an open that occurred at or before the text
   became public.
2. Every ``entry_date`` is a member of the SPY trading-day index.
3. ``entry_date`` is the *first* such trading day -- not merely some later one.
   A rule that skipped ahead would be conservative rather than leaky, but it
   would silently change the horizon being measured.
4. The 2020-03-15 Sunday-evening emergency cut maps to Monday 2020-03-16, and
   the 2008-10-08 07:00 ET pre-open release maps to 2008-10-09. These two cases
   are exactly where naive ``release_date + 1`` arithmetic goes wrong.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Phase 1 not yet implemented (src/align.py)")


def test_entry_strictly_after_release() -> None:
    raise NotImplementedError


def test_entry_is_a_trading_day() -> None:
    raise NotImplementedError


def test_entry_is_the_first_available_trading_day() -> None:
    raise NotImplementedError


def test_known_emergency_meetings_align_correctly() -> None:
    raise NotImplementedError


def test_forward_returns_nan_only_in_the_tail() -> None:
    """NaNs are permitted only for events whose exit date exceeds the price data."""
    raise NotImplementedError
