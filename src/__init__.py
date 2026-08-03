"""fed-sentiment-alpha: does the *change* in FOMC sentiment predict SPY returns?

Module map (mirrors the phases in PLAN.md section 7):

    config.py         typed loader for config.yaml                    (Phase 0)
    scrape_fomc.py    federalreserve.gov crawler -> raw HTML + manifest (Phase 1)
    prices.py         yfinance SPY loader, adjusted OHLC              (Phase 1)
    align.py          release_datetime -> next trading open; fwd returns (Phase 1)
    sentiment.py      FinBERT sentence scoring and aggregation         (Phase 2)
    alpha_signal.py   trailing rolling Z-score and the position rule   (Phase 3)
    backtest.py       vectorised PnL, Information Coefficient, CIs     (Phase 4)
    diagnostics.py    IC decay, cost sensitivity, Fed-speak audit      (Phase 4)

Note on `alpha_signal.py`: PLAN.md section 6.1 names this file `signal.py`, but
`signal` is a Python *standard library* module (Unix/Windows process signals)
that `torch` imports at startup. A top-level `signal.py` on ``sys.path`` shadows
it and produces a baffling ImportError deep inside torch. Renaming avoids a
class of bug that costs hours to diagnose.
"""

__version__ = "0.1.0"
