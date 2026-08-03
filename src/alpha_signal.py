"""Phase 3 -- raw sentiment to a tradable signal.

Named ``alpha_signal`` rather than ``signal`` (as in PLAN.md section 6.1)
because ``signal`` is a Python standard-library module that ``torch`` imports at
startup; a top-level ``signal.py`` shadows it and produces a baffling ImportError
from deep inside torch.

The economic idea: markets price surprises, not levels
------------------------------------------------------
Model the observed sentiment as a slow baseline plus an innovation::

    S_t = mu_t + eps_t

``mu_t`` is the Fed's prevailing tone, which drifts across regimes; ``eps_t`` is
this meeting's *departure* from it. Markets have largely priced ``mu_t`` already
-- everyone knows the current stance. What can move prices is ``eps_t``.

Phase 2 measured how large that drift is (PLAN.md section 4.2): mean S_prob runs
+0.168 pre-GFC, +0.039 through ZIRP, +0.133, then +0.055 in the hiking cycle --
swings larger than a within-era standard deviation. So this is not a stylistic
preference. Feeding raw ``S_t`` into an IC would largely measure "which decade is
this", and since markets also trend across decades, that manufactures
correlation out of two shared time trends: a **spurious regression**.

The estimator, and the one line that matters
--------------------------------------------
::

    mu    = S.rolling(L).mean().shift(1)   # .shift(1) => EXCLUDE the current point
    sigma = S.rolling(L).std().shift(1)
    Z     = (S - mu) / sigma

⚠️ **Without ``.shift(1)``, ``S_t`` enters its own mean.** The estimate would be
contaminated by the very observation it is supposed to be surprised by, and the
whole series becomes untradeable -- you cannot know today's mean until today has
happened. This is the textbook leak, and it is one method call away at all times.

``tests/test_zscore_no_lookahead.py`` is the canary: it rebuilds every ``Z_t``
with an independent backward-looking loop that shares no code with this
implementation, and asserts an exact match.

Indexing, spelled out because off-by-one here is invisible
-----------------------------------------------------------
``rolling(L)`` at position *i* covers ``[i-L+1, i]`` and is first defined at
*i = L-1*. ``.shift(1)`` moves each value one row later, so at position *i* the
statistic covers ``[i-L, i-1]`` -- strictly the past -- and is first defined at
*i = L*. Rows ``0 .. L-1`` are therefore NaN by construction, costing 6 of 225
observations at *L* = 6.

``.std()`` uses ``ddof=1`` (the pandas default): the **sample** standard
deviation, dividing by *L*-1 rather than *L*. Correct here, because ``mu`` is
itself estimated from the same window, which consumes one degree of freedom.

The trading rule and the sign
-----------------------------
::

    position = +1  if Z > +theta   (long SPY)
               -1  if Z < -theta   (short SPY)
                0  otherwise       (cash)

⚠️ **The sign is a pre-registered HYPOTHESIS, not a fitted parameter.**
``hypothesis_sign: dovish_surprise_bullish`` means positive Z -> long.

Phase 2 (PLAN.md section 2.7.1) found that FinBERT's extreme sentences all
describe *economic conditions*, never *policy stance* -- so its "positive"
tracks "the economy is strong", which implies tightening, which is a headwind
for equities. That makes a **negative** IC the economically coherent outcome.

That is a *prediction*, and it must not be used to flip the sign now. Fitting a
binary parameter to ~219 observations after seeing the data is overfitting even
when the reasoning sounds principled. The sign stays as pre-registered; Phase 4
tests it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.sentiment import load_sentiment_scores

# Maps the pre-registered hypothesis to the direction of the trade.
HYPOTHESIS_SIGN = {
    "dovish_surprise_bullish": +1,
    "dovish_surprise_bearish": -1,
}


def rolling_zscore(scores: pd.Series, window: int) -> pd.DataFrame:
    """Trailing Z-score that excludes the current observation.

    Returns a frame with ``mu``, ``sigma`` and ``z``, indexed like ``scores``.

    ⚠️ ``scores`` **must already be in chronological order.** ``rolling`` works
    on positions, not on any date index, so an unsorted input silently produces
    a "trailing" window drawn from the future. :func:`build_signal` sorts before
    calling; this function re-checks nothing because it has no date to check
    against -- which is exactly why the caller must be disciplined.
    """
    if window < 2:
        raise ValueError("window must be >= 2; a standard deviation needs two points")

    mu = scores.rolling(window).mean().shift(1)
    sigma = scores.rolling(window).std().shift(1)  # ddof=1, the sample sd

    # A zero sigma means L identical scores in a row: the surprise is undefined,
    # not infinite. Mapping it to NaN drops the observation loudly rather than
    # emitting a +/-inf that would dominate any correlation it touched.
    sigma_safe = sigma.where(sigma > 0)

    return pd.DataFrame({"mu": mu, "sigma": sigma, "z": (scores - mu) / sigma_safe})


def positions_from_z(z: pd.Series, theta: float, sign: int = +1) -> pd.Series:
    """Apply the threshold rule, returning positions in {-1, 0, +1}.

    ``sign`` is +1 under the pre-registered ``dovish_surprise_bullish``
    hypothesis. It is a declared hypothesis, never a free parameter.

    NaN ``z`` (the warm-up rows) maps to **0**, i.e. flat. That is the only
    honest treatment: with no signal there is no position, and a NaN silently
    propagating into a return calculation would quietly drop observations from
    the performance series while leaving them in the IC series.
    """
    if sign not in (+1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign!r}")

    position = pd.Series(0, index=z.index, dtype="int8")
    position[z > theta] = sign
    position[z < -theta] = -sign
    position[z.isna()] = 0
    return position


def signal_sign(cfg: Config) -> int:
    """Resolve the pre-registered hypothesis to +1 or -1."""
    return HYPOTHESIS_SIGN[cfg.signal.hypothesis_sign]


def build_signal(cfg: Config, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge sentiment onto the panel and emit S, mu, sigma, Z, position.

    Writes the augmented ``data/processed/panel.parquet``.

    The merge is asserted to preserve every row. A silent row loss here would
    shrink the sample without anything failing -- see F16, where ``doc_id`` was
    int64 in one cache and str in the other.
    """
    processed = cfg.paths.resolve("data_processed")
    if panel is None:
        panel = pd.read_parquet(processed / "panel.parquet")

    scores = load_sentiment_scores(cfg)

    panel = panel.copy()
    panel["doc_date"] = pd.to_datetime(panel["doc_date"])
    scores["doc_date"] = pd.to_datetime(scores["doc_date"])

    n_before = len(panel)
    keep = ["doc_date", "S_prob", "S_count", "n_sentences", "n_boilerplate", "mean_confidence"]
    merged = panel.merge(scores[keep], on="doc_date", how="left", validate="one_to_one")

    if len(merged) != n_before:
        raise AssertionError(f"merge changed row count: {n_before} -> {len(merged)}")
    missing = int(merged["S_prob"].isna().sum())
    if missing:
        raise AssertionError(
            f"{missing} panel rows have no sentiment score. Every statement in the "
            "panel was scored in Phase 2, so this means the join key drifted."
        )

    # Chronological order is a precondition of `rolling`, which works on
    # positions and cannot detect that its window came from the future.
    merged = merged.sort_values("doc_date").reset_index(drop=True)
    if not merged["doc_date"].is_monotonic_increasing:
        raise AssertionError("panel is not chronologically sorted; rolling would leak")

    # The pre-registered aggregation is the signal; the other is carried
    # alongside for the Phase 4 robustness grid.
    primary = "S_prob" if cfg.sentiment.aggregation == "prob" else "S_count"
    merged["S"] = merged[primary]

    stats = rolling_zscore(merged["S"], cfg.signal.zscore_window_L)
    merged[["mu", "sigma", "Z"]] = stats[["mu", "sigma", "z"]]

    merged["position"] = positions_from_z(
        merged["Z"], cfg.signal.threshold_theta, sign=signal_sign(cfg)
    )

    # Turnover drives transaction cost in Phase 4: you pay when you enter, flip
    # or exit. The first event has no prior position, so its turnover is the
    # cost of establishing whatever position it takes.
    merged["turnover"] = merged["position"].diff().abs()
    merged.loc[merged.index[0], "turnover"] = abs(merged.loc[merged.index[0], "position"])

    _assert_zscore_is_backward_looking(merged, cfg)

    merged.to_parquet(processed / "panel.parquet")
    return merged


def _assert_zscore_is_backward_looking(panel: pd.DataFrame, cfg: Config) -> None:
    """Recompute Z with a naive backward loop and require an exact match.

    Duplicated in the test suite deliberately. A test protects you when you run
    it; a build-time assertion protects you every time the pipeline runs. For
    the one error that would invalidate the entire project, both is correct.

    Note this reimplementation shares no code with :func:`rolling_zscore` -- if
    it used the same ``.rolling().shift()`` call it could not possibly detect a
    missing shift, because both would be wrong identically.
    """
    window = cfg.signal.zscore_window_L
    values = panel["S"].to_numpy()

    for i in range(len(panel)):
        recorded = panel["Z"].iloc[i]
        if i < window:
            if not np.isnan(recorded):
                raise AssertionError(f"row {i} < L={window} must have NaN Z, got {recorded}")
            continue

        past = values[i - window : i]  # strictly BEFORE i
        expected_mu = past.mean()
        expected_sigma = past.std(ddof=1)
        expected = (values[i] - expected_mu) / expected_sigma if expected_sigma > 0 else np.nan

        if np.isnan(expected) and np.isnan(recorded):
            continue
        if not np.isclose(recorded, expected, rtol=1e-10, atol=1e-12):
            raise AssertionError(
                f"LOOK-AHEAD DETECTED at row {i} ({panel['doc_date'].iloc[i]}): "
                f"recorded Z={recorded!r} but a strictly-backward window gives "
                f"{expected!r}. The .shift(1) is missing or misapplied."
            )


def summarise_signal(panel: pd.DataFrame, cfg: Config) -> dict:
    """Facts about the produced signal, for the phase report.

    Deliberately contains **no return data**. Phase 3 must not look at
    performance; doing so before the pre-registered evaluation would undermine
    the whole point of pre-registering it.
    """
    valid = panel["Z"].notna()
    positions = panel["position"]

    return {
        "n_events": len(panel),
        "n_warmup_nan": int((~valid).sum()),
        "n_usable": int(valid.sum()),
        "z_mean": float(panel.loc[valid, "Z"].mean()),
        "z_std": float(panel.loc[valid, "Z"].std()),
        "z_min": float(panel.loc[valid, "Z"].min()),
        "z_max": float(panel.loc[valid, "Z"].max()),
        "n_long": int((positions == 1).sum()),
        "n_short": int((positions == -1).sum()),
        "n_flat": int((positions == 0).sum()),
        "pct_in_market": round(float((positions != 0).mean()) * 100, 1),
        "total_turnover": float(panel["turnover"].sum()),
        "mean_turnover_per_event": round(float(panel["turnover"].mean()), 3),
    }
