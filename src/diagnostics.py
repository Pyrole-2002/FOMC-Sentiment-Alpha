"""Phase 4 -- the three research-grade diagnostics (PLAN.md section 8).

These are what separate "a backtest" from "research". Each answers a question a
research desk would actually ask.

1. IC decay -- when does the market price this in?
2. Cost sensitivity -- does the edge survive being harvested?
3. Fed-speak audit -- does the model actually understand the text?
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config


def plot_ic_decay(ic_table: pd.DataFrame, out_path: Path) -> Path:
    """Bar chart of Spearman IC by horizon, with bootstrap CIs as error bars.

    **Alpha half-life** is how fast predictive power fades as the horizon
    lengthens; the shape reveals how quickly the market absorbs the information.
    Monotone decay is the economically sensible pattern. A *flat* line invites
    the question "are your forward returns overlapping and autocorrelated?" --
    which ``align.check_overlap`` will already have answered.
    """
    raise NotImplementedError("Phase 4")


def plot_cost_sensitivity(panel: pd.DataFrame, cfg: Config, out_path: Path) -> Path:
    """Strategy Sharpe as per-trade cost rises across ``cfg.costs.bps_grid``.

    A signal is only alpha if it survives the cost of harvesting it. This curve
    is the go/no-go test for real-world viability, and even a few bps can flip a
    marginal strategy negative.
    """
    raise NotImplementedError("Phase 4")


def extreme_sentences(sentence_scores: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """The k most positive and k most negative sentences FinBERT found.

    Read them as a human: do they *look* dovish and hawkish? This is a sanity
    check on the black box, and it costs nothing because sentence-level scores
    were cached in Phase 2.
    """
    raise NotImplementedError("Phase 4")


def fedspeak_confusion_matrix(labelled: pd.DataFrame) -> pd.DataFrame:
    """Compare FinBERT's labels against a hand-labelled sample of Fed sentences.

    Roughly 30 sentences labelled by hand as hawk/dove/neutral, scored with
    ``sklearn.metrics.confusion_matrix``. This quantifies FinBERT's agreement
    with domain judgement on *central-bank* language specifically -- the model
    was trained on analyst news, so the transfer is a genuine open question.

    If FinBERT systematically mislabels hedged Fed prose, **that is a finding**,
    and it explains a weak IC rather than being explained away by one.
    """
    raise NotImplementedError("Phase 4")
