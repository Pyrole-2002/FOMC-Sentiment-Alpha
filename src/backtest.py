"""Phase 4 -- vectorised backtest and the Information Coefficient.

Primary metric: the IC, not the Sharpe ratio
--------------------------------------------
**Information Coefficient (IC)** is the correlation between the signal at time
*t* and the realised return over the following *h*::

    IC = corr(signal_t, R_{t -> t+h})

**Spearman (rank) IC** is primary: it correlates *ranks*, asking only "did a
higher signal go with a higher return?", which is robust to the fat tails and
outliers endemic to return data. **Pearson IC** is a secondary cross-check.

Scale, which surprises people: IC ~ 0.05 is a strong, institution-grade signal
in cross-sectional equity research. Above 0.10 in a clean large sample is
exceptional and should prompt a bug hunt.

IC is preferred over Sharpe because it measures predictive correlation directly,
decoupled from the position-sizing and leverage choices that make Sharpe so easy
to inflate.

Error bars are mandatory, not optional
--------------------------------------
With n ~ 250 events, a point estimate alone is meaningless.

**t-statistic** -- how many standard errors the estimate sits from zero::

    t = IC * sqrt((n - 2) / (1 - IC^2))

At n=200, an IC of 0.06 gives t ~ 0.85: **not significant**. That arithmetic is
sobering and honest, and it should be shown rather than buried.

**Bootstrap confidence interval** -- resample the (signal, forward-return) pairs
with replacement thousands of times, recompute the IC each time, and report the
2.5th-97.5th percentiles. It assumes nothing about normality, which makes it the
most defensible error bar for a small, fat-tailed sample.

Transaction costs
-----------------
A **basis point (bp)** is 0.01% = 0.0001. Cost per event is
``bps * |position_t - position_{t-1}|`` -- you pay when you enter, flip, or
exit. Most student backtests omit costs and are therefore fiction.

Honesty clause (PLAN.md section 5.5)
------------------------------------
If the IC confidence interval straddles zero, the correct deliverable is
"no statistically significant edge detected at these horizons, given n ~ 250;
here is the interval, and here is the sample size needed to detect an effect of
plausible magnitude." Do **not** tune the pipeline until a positive appears --
that process *is* the overfitting this project is designed to avoid.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config


def information_coefficient(
    signal: pd.Series, returns: pd.Series, method: str = "spearman"
) -> dict:
    """Compute the IC with its t-statistic, p-value, and sample size.

    Pairs with a NaN on either side are dropped (and the count reported), so
    ``n`` always reflects the observations actually used.
    """
    raise NotImplementedError("Phase 4")


def bootstrap_ic_ci(
    signal: pd.Series,
    returns: pd.Series,
    n_iter: int,
    ci: float,
    seed: int,
    method: str = "spearman",
) -> dict:
    """Percentile bootstrap confidence interval for the IC.

    Resamples (signal, return) **pairs** with replacement -- resampling the two
    series independently would destroy the very association being measured.
    """
    raise NotImplementedError("Phase 4")


def ic_grid(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """IC across every horizon and both aggregations -- the full grid, not the max.

    Reporting the whole grid is the defence against cherry-picking (PLAN.md 5.1):
    a signal positive across many horizons is credible; one that spikes at a
    single lucky *h* is not.
    """
    raise NotImplementedError("Phase 4")


def strategy_returns(
    positions: pd.Series, forward_returns: pd.Series, cost_bps: float = 0.0
) -> pd.Series:
    """Per-event strategy return, net of turnover cost."""
    raise NotImplementedError("Phase 4")


def performance_metrics(strat_returns: pd.Series, periods_per_year: float = 8.0) -> dict:
    """Headline metrics, hand-computed so the report never depends on quantstats.

    Returns CAGR, annualised volatility, Sharpe, max drawdown, win rate,
    average win/loss, and the number of trades.

    ``periods_per_year`` defaults to 8 because FOMC events, not calendar days,
    are the observation unit -- annualising an event-driven series as if it were
    daily would inflate the Sharpe by roughly sqrt(252/8) ~ 5.6x.
    """
    raise NotImplementedError("Phase 4")
