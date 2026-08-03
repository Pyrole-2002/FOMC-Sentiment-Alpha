"""Phase 3 leak canary: the Z-score must not see its own observation.

Currently skipped -- ``rolling_zscore`` lands in Phase 3.

The central test rebuilds every ``Z_t`` with a naive Python loop over rows
**strictly before** *t* and asserts an exact match against the vectorised
column. Written independently of the implementation on purpose: if both used
``.rolling().shift()`` the test could not detect a missing shift.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Phase 3 not yet implemented (src/alpha_signal.py)")


def test_zscore_matches_naive_backward_looking_reconstruction() -> None:
    """The canary. Loop-recompute mu/sigma from rows < t and compare exactly."""
    raise NotImplementedError


def test_first_L_observations_are_nan() -> None:
    """With .shift(1) the window at index i covers [i-L, i-1]: NaN until i = L."""
    raise NotImplementedError


def test_future_values_do_not_change_past_zscores() -> None:
    """Perturbing S_t must leave every Z_k for k <= t untouched.

    The sharpest formulation of causality: appending or altering future data
    cannot retroactively change a past signal.
    """
    raise NotImplementedError


def test_positions_respect_threshold() -> None:
    """|Z| <= theta => position 0; sign of position follows sign of Z."""
    raise NotImplementedError
