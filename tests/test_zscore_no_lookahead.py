"""Phase 3 leak canary: the Z-score must not see its own observation.

This is the most important test file in the project. The look-ahead it guards
against is one method call away at all times -- dropping ``.shift(1)`` from
``rolling_zscore`` produces a series that still looks entirely plausible,
trades sensibly, and is completely untradeable in reality.

🧠 **Why the reconstruction is written as a naive Python loop.** If this test
computed the expected value with ``.rolling().shift()`` -- the same call the
implementation uses -- it could not possibly detect a missing shift, because
both would be wrong identically. A test must be able to fail for the bug it is
named after, which means it has to reach the answer by a different route. The
loop is slow and ugly on purpose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.alpha_signal import (
    HYPOTHESIS_SIGN,
    positions_from_z,
    rolling_zscore,
    signal_sign,
)
from src.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def series() -> pd.Series:
    """A deterministic pseudo-random series; no RNG seeding dependence."""
    values = np.sin(np.arange(40) * 0.7) * 0.2 + np.cos(np.arange(40) * 0.31) * 0.1
    return pd.Series(values)


def _naive_zscore(values: np.ndarray, window: int) -> np.ndarray:
    """Recompute Z with an explicit backward loop. Shares no code with src."""
    out = np.full(len(values), np.nan)
    for i in range(window, len(values)):
        past = values[i - window : i]  # strictly BEFORE i
        sigma = past.std(ddof=1)
        if sigma > 0:
            out[i] = (values[i] - past.mean()) / sigma
    return out


# ---------------------------------------------------------------------------
# The canary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window", [2, 3, 6, 12])
def test_zscore_matches_naive_backward_reconstruction(series, window):
    """THE canary. Every Z_t must be reproducible from rows strictly before t."""
    produced = rolling_zscore(series, window)["z"].to_numpy()
    expected = _naive_zscore(series.to_numpy(), window)
    np.testing.assert_allclose(produced, expected, rtol=1e-12, equal_nan=True)


def test_first_L_observations_are_nan(series):
    """With .shift(1) the window at index i covers [i-L, i-1]: NaN until i = L."""
    window = 6
    z = rolling_zscore(series, window)["z"]
    assert z.iloc[:window].isna().all(), "warm-up rows must be NaN"
    assert z.iloc[window:].notna().all(), "everything after warm-up must be defined"


def test_future_values_do_not_change_past_zscores(series):
    """The sharpest statement of causality.

    Appending or altering FUTURE data must leave every past Z untouched. A
    forward-looking window would silently rewrite history here -- and this is
    the formulation that catches a centred or expanding window, which the
    reconstruction test alone might not if it shared the same off-by-one.
    """
    window = 6
    cut = 25
    full = rolling_zscore(series, window)["z"]

    perturbed = series.copy()
    perturbed.iloc[cut:] = perturbed.iloc[cut:] * -3.0 + 99.0
    after = rolling_zscore(perturbed, window)["z"]

    # Z at index i depends on rows [i-L, i], so indices strictly below `cut`
    # must be untouched by anything happening at or after `cut`.
    pd.testing.assert_series_equal(full.iloc[:cut], after.iloc[:cut])


def test_appending_data_does_not_change_existing_zscores(series):
    """Same guarantee, expressed as a streaming/online property."""
    window = 6
    short = rolling_zscore(series.iloc[:30], window)["z"]
    long = rolling_zscore(series, window)["z"]
    pd.testing.assert_series_equal(short, long.iloc[:30])


# ---------------------------------------------------------------------------
# Estimator details that are easy to get subtly wrong
# ---------------------------------------------------------------------------


def test_window_uses_sample_standard_deviation():
    """ddof=1, because mu is estimated from the same window (one d.o.f. spent)."""
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0])
    out = rolling_zscore(values, 6)
    past = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert out["sigma"].iloc[6] == pytest.approx(past.std(ddof=1))
    assert out["mu"].iloc[6] == pytest.approx(past.mean())
    assert out["z"].iloc[6] == pytest.approx((10.0 - past.mean()) / past.std(ddof=1))


def test_zero_sigma_yields_nan_not_infinity():
    """L identical values leave the surprise UNDEFINED, not infinite.

    An inf would propagate into any correlation it touched and dominate it
    entirely, which is a silent catastrophe rather than a loud failure.
    """
    values = pd.Series([0.5] * 6 + [0.9])
    out = rolling_zscore(values, 6)
    assert out["sigma"].iloc[6] == 0.0
    assert np.isnan(out["z"].iloc[6])
    assert not np.isinf(out["z"]).any()


def test_window_below_two_is_rejected():
    with pytest.raises(ValueError, match="window must be >= 2"):
        rolling_zscore(pd.Series([1.0, 2.0, 3.0]), 1)


# ---------------------------------------------------------------------------
# The trading rule
# ---------------------------------------------------------------------------


def test_positions_respect_threshold():
    z = pd.Series([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    pos = positions_from_z(z, theta=1.0, sign=+1)
    # Strictly greater / strictly less: |Z| == theta is NOT a trade.
    assert pos.tolist() == [-1, -1, 0, 0, 0, 0, 0, 1, 1]


def test_positions_follow_the_sign_convention():
    z = pd.Series([2.0, -2.0])
    assert positions_from_z(z, 1.0, sign=+1).tolist() == [1, -1]
    assert positions_from_z(z, 1.0, sign=-1).tolist() == [-1, 1]


def test_nan_z_is_flat_not_nan():
    """No signal means no position. A NaN position would silently drop the row
    from the performance series while leaving it in the IC series."""
    pos = positions_from_z(pd.Series([np.nan, 2.0]), 1.0, sign=+1)
    assert pos.tolist() == [0, 1]
    assert pos.dtype.kind in "iu"


def test_invalid_sign_is_rejected():
    with pytest.raises(ValueError, match="sign must be"):
        positions_from_z(pd.Series([1.0]), 1.0, sign=0)


def test_hypothesis_sign_resolves_as_preregistered(cfg):
    """The pre-registered hypothesis is dovish_surprise_bullish => +1.

    ⚠️ Phase 2 (PLAN.md 2.7.1) predicts this sign is economically WRONG,
    because FinBERT tracks economic conditions rather than policy stance. That
    prediction must NOT be used to flip the sign: fitting a binary parameter to
    the data after inspecting it is overfitting, however principled it sounds.
    This test pins the sign so a "fix" cannot land quietly.
    """
    assert cfg.signal.hypothesis_sign == "dovish_surprise_bullish"
    assert signal_sign(cfg) == +1
    assert set(HYPOTHESIS_SIGN) == {"dovish_surprise_bullish", "dovish_surprise_bearish"}


# ---------------------------------------------------------------------------
# Integration -- the real panel
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel(cfg):
    path = cfg.paths.resolve("data_processed") / "panel.parquet"
    if not path.exists():
        pytest.skip("panel.parquet not built; run scripts/build_signal.py")
    frame = pd.read_parquet(path)
    if "Z" not in frame.columns:
        pytest.skip("signal not built; run scripts/build_signal.py")
    return frame


def test_real_panel_is_chronological(panel):
    """`rolling` works on positions, so an unsorted panel would leak silently."""
    assert panel["doc_date"].is_monotonic_increasing


def test_real_zscore_matches_naive_reconstruction(panel, cfg):
    """The canary, on the actual data rather than a synthetic series."""
    expected = _naive_zscore(panel["S"].to_numpy(), cfg.signal.zscore_window_L)
    np.testing.assert_allclose(panel["Z"].to_numpy(), expected, rtol=1e-10, equal_nan=True)


def test_real_warmup_rows_are_flat(panel, cfg):
    warmup = panel.head(cfg.signal.zscore_window_L)
    assert warmup["Z"].isna().all()
    assert (warmup["position"] == 0).all()


def test_real_positions_match_the_rule(panel, cfg):
    expected = positions_from_z(panel["Z"], cfg.signal.threshold_theta, sign=signal_sign(cfg))
    assert (panel["position"] == expected).all()


def test_real_z_has_no_infinities(panel):
    assert np.isfinite(panel.loc[panel["Z"].notna(), "Z"]).all()


def test_turnover_is_consistent_with_positions(panel):
    """Turnover drives Phase 4's cost model, so it must reconcile exactly."""
    expected = panel["position"].diff().abs()
    expected.iloc[0] = abs(panel["position"].iloc[0])
    pd.testing.assert_series_equal(
        panel["turnover"].astype(float), expected.astype(float), check_names=False
    )


def test_signal_column_matches_the_preregistered_aggregation(panel, cfg):
    primary = "S_prob" if cfg.sentiment.aggregation == "prob" else "S_count"
    pd.testing.assert_series_equal(panel["S"], panel[primary], check_names=False)
