"""Phase 2 -- FinBERT sentence scoring and document aggregation.

**FinBERT** is BERT fine-tuned on financial text (checkpoint ``ProsusAI/finbert``,
trained on the Financial PhraseBank). For a span of text it emits three
probabilities: P(positive), P(negative), P(neutral).

The 512-token limit
-------------------
A **token** is a sub-word unit produced by the tokenizer; BERT's positional
embedding table has exactly 512 rows, so it physically cannot attend to a longer
sequence. An FOMC statement is several paragraphs -- far beyond that -- so we
split into sentences, score each independently, and aggregate.

Two aggregations (PLAN.md section 4.1), both computed
-----------------------------------------------------
Count-based, ranges in [-1, 1]::

    S_count = (N_positive - N_negative) / N_total

Probability-weighted, more information-dense::

    S_prob = mean(p_pos_i - p_neg_i)

``S_count`` discards confidence -- a 51%-positive sentence counts the same as a
99%-positive one. ``S_prob`` keeps it, but is more exposed to FinBERT being
miscalibrated on central-bank prose. Which is better is a **research finding**,
so both are computed and the choice is pre-registered in ``config.yaml``
(primary = ``prob``) *before* returns are examined.

Label ordering -- a silent sign-flip waiting to happen
------------------------------------------------------
``ProsusAI/finbert`` uses ``id2label = {0: 'positive', 1: 'negative',
2: 'neutral'}``. This is **not** alphabetical and **not** the
negative/neutral/positive ordering most sentiment checkpoints use. Hardcoding
``logits[:, 0]`` as "negative" -- a natural guess -- silently inverts the entire
signal, which would show up only as an unexplained sign flip in the IC.

Always read the mapping from ``model.config.id2label`` at load time and index by
*name*. ``tests/test_sentiment_shapes.py`` asserts the expected mapping so a
future checkpoint revision cannot change it unnoticed.

The out-of-domain caveat
------------------------
FinBERT was trained on analyst and news language, not Fed prose. "The Committee
remains highly attentive to inflation risks" is hedged, euphemistic, and
structurally unlike a news headline. Whether FinBERT's notion of sentiment
transfers is an open question, audited in :mod:`src.diagnostics` (PLAN.md 8.3)
rather than assumed.

That exact sentence, scored during Phase 0 environment validation, returns
``p_neutral=0.64, p_positive=0.34, p_negative=0.02`` -- FinBERT reads a
*hawkish* inflation-vigilance statement as mildly **positive**. That is a
single anecdote, not evidence, but it is a concrete early illustration of the
sign trap in PLAN.md section 2.7: FinBERT's "positive" tracks *sentiment*, not
*equity-directional implication*.

Caching
-------
This module is the only expensive step. Its output is cached so Phases 3-4
iterate instantly, and ``model.eval()`` + ``torch.no_grad()`` + a fixed seed make
re-runs bit-identical. Sentence-level scores are persisted too -- the Fed-speak
audit needs the most extreme individual sentences, and re-running FinBERT to
recover them would be wasteful.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config


def resolve_device(cfg: Config) -> str:
    """Resolve ``cfg.sentiment.device`` to a concrete torch device string.

    ``"auto"`` selects ``"cuda"`` when available, else ``"cpu"``. Keeping this
    device-agnostic means the repo runs unchanged on a reviewer's laptop.

    CUDA note for RTX 50-series (Blackwell, compute capability ``sm_120``):
    only PyTorch builds from the **cu128** index or newer contain sm_120
    kernels. A cu126-or-earlier wheel raises "no kernel image is available for
    execution on the device". See ``scripts/check_gpu.py``.
    """
    raise NotImplementedError("Phase 2")


def split_sentences(text: str) -> list[str]:
    """Split a document into sentences.

    Uses ``nltk.tokenize.sent_tokenize`` (the ``punkt`` model), falling back to
    a regex splitter if the punkt data is unavailable -- so the pipeline never
    hard-depends on a runtime model download.
    """
    raise NotImplementedError("Phase 2")


def score_sentences(sentences: list[str], cfg: Config) -> pd.DataFrame:
    """Score sentences in batches under ``torch.no_grad()``.

    Returns one row per sentence: ``p_pos``, ``p_neg``, ``p_neu``, ``label``
    (arg-max class), ``n_tokens``.
    """
    raise NotImplementedError("Phase 2")


def aggregate(sentence_scores: pd.DataFrame) -> dict:
    """Collapse sentence scores to one document score, both formulas.

    Returns ``S_count``, ``S_prob``, ``n_sentences``, ``n_pos``, ``n_neg``,
    ``n_neu``, ``mean_confidence``.
    """
    raise NotImplementedError("Phase 2")


def strip_boilerplate(sentences: list[str]) -> list[str]:
    """Remove procedural, sentiment-free sentences (Phase 2 stretch goal).

    FOMC statements always close with vote tallies -- "Voting for the monetary
    policy action were..." -- which carry no policy tone but dilute both
    aggregations toward zero. Off by default (``strip_boilerplate: false``) so
    the primary result uses the unmodified text; its effect is reported as a
    robustness check.
    """
    raise NotImplementedError("Phase 2")


def run_corpus(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the whole corpus once and write both caches.

    Writes ``data/processed/sentiment_scores.csv`` (one row per document) and
    ``data/processed/sentence_scores.parquet`` (one row per sentence).
    """
    raise NotImplementedError("Phase 2")
