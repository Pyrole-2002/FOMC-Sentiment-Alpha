"""Phase 3 -- raw sentiment to a tradable signal.

Named ``alpha_signal`` rather than ``signal`` (as in PLAN.md section 6.1)
because ``signal`` is a Python standard-library module that ``torch`` imports at
startup; a top-level ``signal.py`` shadows it and produces a baffling ImportError
from deep inside torch.

The economic idea: markets price surprises, not levels
------------------------------------------------------
The Fed's baseline tone is structurally cautious and hedged. What moves markets
is a statement being more or less hawkish **than recently**. So we normalise
each raw score against its own trailing history::

    Z_t = (S_t - mu_t) / sigma_t

where ``mu_t``, ``sigma_t`` are the mean and standard deviation of *S* over the
previous *L* meetings.

The look-ahead trap (PLAN.md section 4.2)
-----------------------------------------
``mu`` and ``sigma`` must come from a trailing window that **excludes the
current observation**::

    mu    = S.rolling(L).mean().shift(1)   # .shift(1) => drop the current point
    sigma = S.rolling(L).std().shift(1)
    Z     = (S - mu) / sigma

Omitting ``.shift(1)`` lets ``S_t`` contribute to its own mean -- a textbook
leak. ``tests/test_zscore_no_lookahead.py`` is the canary: it rebuilds every
``Z_t`` from rows strictly before *t* and asserts an exact match against the
vectorised column.

Consequence: the first *L* meetings have an undefined window and yield NaN.
They are discarded, costing ~6 of ~250 observations. Worth it.

The trading rule and the sign
-----------------------------
Deliberately simple -- complexity here just invites overfitting::

    position = +1  if Z >  +theta   (long SPY)
               -1  if Z <  -theta   (short SPY)
                0  otherwise        (cash)

**Hawkish** means leaning toward tighter policy (higher rates); generally a
headwind for equities. **Dovish** means leaning toward easier policy; generally
a tailwind.

The sign trap: FinBERT's "positive" is about *sentiment*, not about *equities*.
A hawkish statement may speak positively about a strong economy -- which is
bearish for stocks because it implies hikes. So the sign is taken from the
pre-registered hypothesis (``dovish_surprise_bullish``) and then **tested**; it
is never fitted to whichever direction maximises return, which would be
overfitting a binary parameter on ~250 points.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config


def rolling_zscore(scores: pd.Series, window: int) -> pd.DataFrame:
    """Trailing Z-score excluding the current observation.

    Returns ``mu``, ``sigma``, ``z``. The first ``window`` entries are NaN by
    construction: with ``.shift(1)`` the window at position *i* covers
    ``[i-window, i-1]``, which is only fully populated from *i = window* onward.
    """
    raise NotImplementedError("Phase 3")


def positions_from_z(z: pd.Series, theta: float, sign: int = +1) -> pd.Series:
    """Apply the threshold rule, returning positions in {-1, 0, +1}.

    ``sign`` is +1 under the ``dovish_surprise_bullish`` hypothesis. It is a
    declared hypothesis, not a free parameter.
    """
    raise NotImplementedError("Phase 3")


def build_signal(cfg: Config) -> pd.DataFrame:
    """Merge sentiment onto the panel and emit ``S``, ``mu``, ``sigma``, ``Z``, ``position``.

    Augments ``data/processed/panel.parquet`` in place (as a new file version).
    """
    raise NotImplementedError("Phase 3")
