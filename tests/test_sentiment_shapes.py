"""Phase 2 shape and determinism checks for the FinBERT engine.

Currently skipped -- ``src/sentiment.py`` lands in Phase 2.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Phase 2 not yet implemented (src/sentiment.py)")


def test_finbert_label_ordering_is_as_expected() -> None:
    """Pin ``id2label = {0: positive, 1: negative, 2: neutral}``.

    This ordering is neither alphabetical nor the common
    negative/neutral/positive convention. Indexing logits by position instead of
    by name would silently invert the whole signal. Verified against the live
    checkpoint during Phase 0; asserted here so a future checkpoint revision
    cannot change it unnoticed.
    """
    raise NotImplementedError


def test_probabilities_sum_to_one() -> None:
    """p_pos + p_neg + p_neu == 1 for every sentence (softmax invariant)."""
    raise NotImplementedError


def test_scores_within_bounds() -> None:
    """Both S_count and S_prob lie in [-1, 1] by construction."""
    raise NotImplementedError


def test_no_document_is_lost() -> None:
    """Every input document appears exactly once in the output cache."""
    raise NotImplementedError


def test_sentence_splitting_has_regex_fallback() -> None:
    """Splitting must work with nltk punkt data unavailable."""
    raise NotImplementedError


@pytest.mark.slow
def test_scoring_is_deterministic() -> None:
    """Same input bytes => bit-identical scores, across runs and devices.

    Guaranteed by ``model.eval()``, ``torch.no_grad()``, and a fixed seed.
    Reproducibility is a research value, not just an engineering nicety.
    """
    raise NotImplementedError
