"""Phase 1 -- SPY daily price loader.

**SPY** is the SPDR S&P 500 ETF Trust, the most liquid proxy for US large-cap
equities. It began trading 1993-01-22, which conveniently pre-dates the first
regular FOMC statement (Feb 1994), so the price series never limits the sample.

Price adjustment (PLAN.md section 3.4, with a correction)
---------------------------------------------------------
**Adjustment** means rewriting historical prices so dividends and splits do not
appear as price gaps. SPY distributes ~1.3%/yr across four dividends; unadjusted
prices show each as a phantom one-day loss.

PLAN.md Appendix A sets ``price_field: "Adj Close"`` while section 4.3 asks to
enter at the *adjusted Open*. Those are incompatible: with
``auto_adjust=False``, yfinance returns raw ``Open/High/Low/Close`` **plus** one
extra ``Adj Close`` column -- there is no ``Adj Open``. Entering on a raw Open
and exiting on an ``Adj Close`` mixes two price bases and biases every return.

We therefore use ``auto_adjust=True``, which back-adjusts **all four** OHLC
columns on a consistent basis, so ``Open`` *is* the adjusted open.

Trading calendar
----------------
The **SPY price index is the source of truth** for "is this a trading day."
This is self-consistent: a day we cannot price is a day we cannot trade, by
definition. ``pandas_market_calendars`` is used only to *validate* that index
against the official NYSE calendar and surface any gaps caused by a yfinance
download failure -- a missing day would otherwise shift an entry date by one
and quietly corrupt the panel.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config


def download_prices(cfg: Config) -> pd.DataFrame:
    """Download adjusted daily OHLCV for ``cfg.data.ticker`` and cache to parquet.

    Returns a DataFrame indexed by tz-naive ``date`` (trading days only) with
    columns ``Open/High/Low/Close/Volume``, all split- and dividend-adjusted.

    Cached to ``data/processed/spy_prices.parquet``. yfinance scrapes an
    undocumented endpoint and can rate-limit or change shape without notice, so
    the cache is the working copy and the network is touched only when it is
    missing or explicitly refreshed.
    """
    raise NotImplementedError("Phase 1")


def trading_days(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the sorted index of valid trading days."""
    raise NotImplementedError("Phase 1")


def validate_against_nyse(prices: pd.DataFrame) -> pd.DataFrame:
    """Cross-check the price index against the official NYSE calendar.

    Returns a frame of discrepancies: NYSE sessions absent from the price data
    (download gaps) and price rows on non-NYSE dates (data errors). An empty
    frame is the pass condition.
    """
    raise NotImplementedError("Phase 1")
