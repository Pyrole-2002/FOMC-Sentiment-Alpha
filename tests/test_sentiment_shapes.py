"""Phase 2 shape, correctness and determinism checks for the FinBERT engine.

Layered like the alignment tests: cheap invariant checks on the cached output
(no model load), plus a small number of tests that actually run the model.

The single most important test here is ``test_finbert_label_ordering``. FinBERT
declares ``{0: positive, 1: negative, 2: neutral}`` -- neither alphabetical nor
the common convention -- so indexing logits by position silently inverts the
whole signal without raising anything.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_config
from src.sentiment import (
    LABELS,
    aggregate,
    is_boilerplate,
    load_sentence_scores,
    load_sentiment_scores,
    resolve_device,
    score_sentences,
    split_sentences,
    strip_boilerplate,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def doc_scores(cfg):
    path = cfg.paths.resolve("data_processed") / "sentiment_scores.csv"
    if not path.exists():
        pytest.skip("run scripts/score_sentiment.py first")
    return load_sentiment_scores(cfg)


@pytest.fixture(scope="module")
def sentence_scores(cfg):
    path = cfg.paths.resolve("data_processed") / "sentence_scores.parquet"
    if not path.exists():
        pytest.skip("run scripts/score_sentiment.py first")
    return load_sentence_scores(cfg)


@pytest.fixture(scope="module")
def model_bundle(cfg):
    from src.sentiment import load_model

    return load_model(cfg)


# ---------------------------------------------------------------------------
# The label-ordering canary
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_finbert_label_ordering_is_as_expected(model_bundle):
    """Pin ``id2label = {0: positive, 1: negative, 2: neutral}``.

    Neither alphabetical nor the common negative/neutral/positive convention.
    Indexing logits by position instead of by name would silently invert the
    whole signal. Verified against the live checkpoint; asserted here so a
    future checkpoint revision cannot change it unnoticed.
    """
    _, model, label_index, _ = model_bundle
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    assert id2label == {0: "positive", 1: "negative", 2: "neutral"}
    assert label_index["positive"] == 0
    assert label_index["negative"] == 1
    assert label_index["neutral"] == 2


@pytest.mark.slow
def test_known_sentences_score_in_the_expected_direction(cfg, model_bundle):
    """A behavioural check that would fail loudly if the labels were swapped.

    Deliberately uses unambiguous FINANCIAL-NEWS sentences, the domain FinBERT
    was actually trained on -- not Fed prose, whose correct labelling is the
    open question this project investigates.
    """
    scored = score_sentences(
        [
            "Profits rose sharply and the company raised its full-year guidance.",
            "The company slashed its dividend after a steep decline in revenue.",
        ],
        cfg,
        model_bundle=model_bundle,
    )
    assert scored.loc[0, "p_pos"] > scored.loc[0, "p_neg"], "good news must score positive"
    assert scored.loc[1, "p_neg"] > scored.loc[1, "p_pos"], "bad news must score negative"


# ---------------------------------------------------------------------------
# Invariants on the cached corpus
# ---------------------------------------------------------------------------


def test_probabilities_sum_to_one(sentence_scores):
    """Softmax invariant; tolerance is float32 rounding, not modelling slack."""
    total = sentence_scores[["p_pos", "p_neg", "p_neu"]].sum(axis=1)
    assert (total - 1.0).abs().max() < 1e-5


def test_probabilities_are_in_range(sentence_scores):
    for col in ("p_pos", "p_neg", "p_neu"):
        assert sentence_scores[col].between(0.0, 1.0).all()


def test_label_matches_argmax(sentence_scores):
    """The stored label must be the arg-max class, indexed BY NAME.

    Catches the label-ordering bug from the output side: if columns were mapped
    by position, the recorded label would disagree with the probabilities.
    """
    implied = sentence_scores[["p_pos", "p_neg", "p_neu"]].idxmax(axis=1)
    expected = implied.map({"p_pos": "positive", "p_neg": "negative", "p_neu": "neutral"})
    assert (expected == sentence_scores["label"]).all()


def test_scores_within_bounds(doc_scores):
    """Both aggregations lie in [-1, 1] by construction."""
    assert doc_scores["S_count"].between(-1, 1).all()
    assert doc_scores["S_prob"].between(-1, 1).all()


def test_no_document_is_lost(cfg, doc_scores):
    """Every eligible statement appears exactly once."""
    panel = pd.read_parquet(cfg.paths.resolve("data_processed") / "panel.parquet")
    assert len(doc_scores) == len(panel)
    assert not doc_scores["doc_id"].duplicated().any()
    assert set(doc_scores["doc_date"].astype(str)) == set(panel["doc_date"].astype(str))


def test_no_document_is_empty(doc_scores):
    """A zero-sentence document would aggregate to NaN and vanish downstream."""
    assert (doc_scores["n_sentences"] > 0).all()
    assert doc_scores["S_prob"].notna().all()
    assert doc_scores["S_count"].notna().all()


def test_nothing_was_truncated(cfg, sentence_scores):
    """Hitting the 512-token wall would silently DISCARD text.

    Sentences are short, so this should never fire -- but truncation is a silent
    data loss, and silent data loss is exactly what this project is built to
    refuse.
    """
    assert sentence_scores["n_tokens"].max() < cfg.sentiment.max_length


def test_sentence_counts_reconcile(doc_scores, sentence_scores):
    """The per-document sentence count must match the sentence cache exactly.

    Also an implicit guard on the ``doc_id`` dtype: if the CSV were read with
    default inference, ``doc_id`` would be int64 here and str in the parquet,
    and this join would quietly produce nothing. See ``load_sentiment_scores``.
    """
    per_doc = sentence_scores.groupby("doc_id").size().sort_index()
    recorded = doc_scores.set_index("doc_id")["n_sentences"].sort_index()
    assert per_doc.index.equals(recorded.index), "doc_id keys must align across both caches"
    assert (per_doc.to_numpy() == recorded.to_numpy()).all()


def test_doc_id_is_a_string_in_both_caches(doc_scores, sentence_scores):
    """Pin the dtype that would otherwise cause a silent empty merge."""
    assert doc_scores["doc_id"].map(type).eq(str).all()
    assert sentence_scores["doc_id"].map(type).eq(str).all()


# ---------------------------------------------------------------------------
# Aggregation maths
# ---------------------------------------------------------------------------


def test_aggregate_matches_the_formulas():
    """S_count = (n_pos - n_neg)/N and S_prob = mean(p_pos - p_neg)."""
    frame = pd.DataFrame(
        {
            "p_pos": [0.9, 0.1, 0.2],
            "p_neg": [0.05, 0.8, 0.2],
            "p_neu": [0.05, 0.1, 0.6],
            "label": ["positive", "negative", "neutral"],
            "n_tokens": [10, 12, 8],
            "is_boilerplate": [False, False, False],
        }
    )
    out = aggregate(frame)
    assert out["S_count"] == pytest.approx((1 - 1) / 3)
    assert out["S_prob"] == pytest.approx(((0.9 - 0.05) + (0.1 - 0.8) + (0.2 - 0.2)) / 3)
    assert out["n_sentences"] == 3


def test_aggregate_handles_an_empty_document():
    out = aggregate(
        pd.DataFrame(columns=["p_pos", "p_neg", "p_neu", "label", "n_tokens", "is_boilerplate"])
    )
    assert out["n_sentences"] == 0
    assert pd.isna(out["S_prob"])


# ---------------------------------------------------------------------------
# Sentence splitting and boilerplate
# ---------------------------------------------------------------------------


def test_sentence_splitting_works():
    text = "The Committee decided to hold rates. Inflation remains elevated. Growth has slowed."
    assert len(split_sentences(text)) == 3


def test_sentence_splitting_has_regex_fallback(monkeypatch):
    """Splitting must still work when nltk's punkt data is unavailable.

    Simulated by making the nltk import raise, which is what a proxy-blocked
    download looks like from inside the function.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "nltk":
            raise ImportError("simulated: punkt unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    text = "The Committee decided to hold rates. Inflation remains elevated."
    assert len(split_sentences(text)) == 2


def test_empty_text_yields_no_sentences():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_boilerplate_detection():
    assert is_boilerplate("Voting for the monetary policy action were Jerome H. Powell, Chair;")
    assert is_boilerplate("In taking the discount rate action, the Federal Reserve Board approved")
    assert is_boilerplate("Release Date: February 2, 2000")
    assert not is_boilerplate("The Committee remains highly attentive to inflation risks.")


def test_strip_boilerplate_keeps_policy_text():
    sentences = [
        "The Committee decided to maintain the target range.",
        "Voting for the monetary policy action were Jerome H. Powell, Chair.",
    ]
    kept = strip_boilerplate(sentences)
    assert len(kept) == 1
    assert kept[0].startswith("The Committee decided")


# ---------------------------------------------------------------------------
# Determinism and configuration
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_scoring_is_deterministic(cfg, model_bundle):
    """Same input bytes => bit-identical scores.

    Guaranteed by ``model.eval()`` (no dropout), ``torch.no_grad()``, and a
    fixed seed. Note the guarantee is per-device and per-batch-size:
    floating-point addition is not associative, so a different batch size
    changes reduction order and can move the final bits. Both are pinned in
    config.
    """
    sentences = [
        "The Committee remains highly attentive to inflation risks.",
        "Household spending and business fixed investment have grown strongly.",
    ]
    first = score_sentences(sentences, cfg, model_bundle=model_bundle)
    second = score_sentences(sentences, cfg, model_bundle=model_bundle)
    pd.testing.assert_frame_equal(first, second)


def test_device_resolution(cfg):
    assert resolve_device(cfg) in {"cuda", "cpu"}


def test_labels_constant_matches_the_checkpoint_classes():
    assert set(LABELS) == {"positive", "negative", "neutral"}
