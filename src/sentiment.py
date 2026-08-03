"""Phase 2 -- FinBERT sentence scoring and document aggregation.

**FinBERT** is BERT fine-tuned on financial text (checkpoint ``ProsusAI/finbert``,
trained on the Financial PhraseBank). For a span of text it emits three
probabilities: P(positive), P(negative), P(neutral).

The 512-token limit
-------------------
A **token** is a sub-word unit produced by the tokenizer. BERT's positional
embedding table has exactly 512 rows, so position 513 does not exist -- this is
an *architectural* wall, not a performance guideline. An FOMC statement runs far
beyond it, so we split into sentences, score each independently, and aggregate.

⚠️ Label ordering -- a silent sign-flip waiting to happen
---------------------------------------------------------
``ProsusAI/finbert`` declares ``id2label = {0: 'positive', 1: 'negative',
2: 'neutral'}``. This is **neither alphabetical** nor the
negative/neutral/positive ordering most sentiment checkpoints use. Writing
``probs[:, 0]`` and calling it "negative" -- the natural guess, and what most
tutorial code does -- silently inverts the entire signal. Nothing crashes; you
simply get an IC with the wrong sign.

We therefore resolve column indices **by name** from ``model.config.id2label``
at load time, and ``tests/test_sentiment_shapes.py`` pins the mapping so a
future checkpoint revision cannot change it unnoticed.

Two aggregations (PLAN.md section 4.1), both computed
-----------------------------------------------------
Count-based, in [-1, 1]::

    S_count = (N_positive - N_negative) / N_total

Probability-weighted::

    S_prob = mean(p_pos_i - p_neg_i)

``S_count`` discards confidence -- a 51%-positive sentence counts the same as a
99%-positive one. ``S_prob`` keeps it, but is more exposed to FinBERT being
**miscalibrated** on central-bank prose (a *calibrated* model is one whose
stated 70% confidence is right ~70% of the time; neural classifiers are
typically over-confident, and more so out of domain).

Which is better is a genuine research question, so both are computed and the
choice is pre-registered in ``config.yaml`` (primary = ``prob``) **before**
returns are examined.

The out-of-domain caveat
------------------------
FinBERT was trained on analyst and news language, not Fed prose. Scored during
Phase 0 environment validation::

    "The Committee remains highly attentive to inflation risks."
    -> p_neutral=0.642, p_positive=0.337, p_negative=0.021

FinBERT reads a *hawkish* inflation-vigilance statement as mildly **positive**.
That is one anecdote, not evidence, but it is a concrete instance of the sign
trap in PLAN.md section 2.7: FinBERT's axis is *sentiment valence*, not
*equity-directional implication*, and those are not the same axis. Section 8.3
quantifies the gap rather than assuming it away.

Determinism
-----------
``model.eval()`` disables dropout; ``torch.no_grad()`` disables autograd; the
seed is fixed. Inference is then deterministic **for a fixed device and batch
size** -- floating-point reductions are not associative, so changing the batch
size changes summation order and can move the last bits. Both are pinned in
config, and ``test_scoring_is_deterministic`` verifies re-runs match.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import Config

# The three class names FinBERT declares. We look these up by name; the ORDER
# here is deliberately irrelevant to the code.
LABELS = ("positive", "negative", "neutral")

# FOMC statements close with a vote tally: "Voting for the monetary policy
# action were: ..." followed by a list of names, and sometimes a dissent
# sentence. These carry no policy tone but are long, so they dilute both
# aggregations toward zero. Detected here; stripping is OFF by default
# (config `strip_boilerplate: false`) so the primary result uses unmodified
# text and the effect is reported as a robustness check.
_BOILERPLATE_RE = re.compile(
    r"^\s*("
    r"voting (for|against) the "
    r"|voting for the fomc monetary policy action"
    r"|in taking the discount rate action"
    r"|in a related action, the board of governors (approved|voted)"
    r"|the board of governors of the federal reserve system (approved|voted)"
    r"|absent and not voting"
    r"|release date:"
    r"|for (immediate )?release"
    r")",
    re.IGNORECASE,
)

# Fallback sentence splitter for when nltk's `punkt` data is unavailable.
# Splits on ., ! or ? followed by whitespace and a capital letter or digit.
# Deliberately conservative: over-splitting a sentence is harmless here (each
# fragment still gets scored), whereas under-splitting risks exceeding 512
# tokens, and truncation would silently discard text.
_REGEX_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def resolve_device(cfg: Config) -> str:
    """Resolve ``cfg.sentiment.device`` to a concrete torch device string.

    ``"auto"`` selects ``"cuda"`` when available, else ``"cpu"``. Keeping this
    device-agnostic means the repository runs unchanged on a reviewer's laptop.

    CUDA note for RTX 50-series (Blackwell, compute capability ``sm_120``): only
    PyTorch builds from the **cu128** index or newer contain sm_120 kernels. An
    earlier wheel installs cleanly, reports ``cuda.is_available() == True``, and
    then fails at the first matmul. See ``scripts/check_gpu.py``.
    """
    import torch

    if cfg.sentiment.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.sentiment.device


def load_model(cfg: Config):
    """Load FinBERT and return ``(tokenizer, model, label_index, device)``.

    ``label_index`` maps each class NAME to its column position in the model's
    output, read from ``model.config.id2label``. Every downstream indexing
    operation uses this map -- never a hardcoded position.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(cfg.sentiment.seed)

    device = resolve_device(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.sentiment.model)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.sentiment.model)
    model.eval().to(device)

    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    label_index = {name: idx for idx, name in id2label.items()}

    missing = set(LABELS) - set(label_index)
    if missing:
        raise RuntimeError(
            f"checkpoint {cfg.sentiment.model!r} does not expose {missing}; "
            f"it declares {id2label}. Indexing by position would silently "
            "produce a wrongly-signed signal, so we refuse to continue."
        )
    return tokenizer, model, label_index, device


def split_sentences(text: str) -> list[str]:
    """Split a document into sentences.

    Uses ``nltk.tokenize.sent_tokenize`` (the ``punkt`` model) when available,
    falling back to a regex splitter otherwise -- so the pipeline never
    hard-depends on a runtime model download that can fail behind a proxy.
    """
    stripped = text.strip()
    if not stripped:
        return []

    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
        sentences = nltk.tokenize.sent_tokenize(stripped)
    except Exception:
        sentences = _REGEX_SPLIT.split(stripped)

    # Documents arrive newline-separated by paragraph; sentence tokenizers keep
    # those newlines inside a "sentence". Flatten so each line is considered.
    flat: list[str] = []
    for sentence in sentences:
        flat.extend(part.strip() for part in sentence.split("\n"))

    # Drop fragments too short to carry sentiment (stray headers, list bullets).
    return [s for s in flat if len(s) >= 15]


def is_boilerplate(sentence: str) -> bool:
    """True for procedural, sentiment-free sentences (vote tallies, headers)."""
    return bool(_BOILERPLATE_RE.match(sentence))


def strip_boilerplate(sentences: list[str]) -> list[str]:
    """Remove procedural sentences (PLAN.md section 4.1, stretch goal).

    Off by default. FOMC statements always close with vote tallies -- "Voting
    for the monetary policy action were..." -- which carry no policy tone but
    are long, so they dilute both aggregations toward zero. Because statement
    length grew markedly over the sample, the dilution is *era-correlated*,
    which makes it a potential confound rather than mere noise.
    """
    return [s for s in sentences if not is_boilerplate(s)]


def score_sentences(sentences: list[str], cfg: Config, model_bundle=None) -> pd.DataFrame:
    """Score sentences in batches under ``torch.no_grad()``.

    Returns one row per sentence: ``sentence``, ``p_pos``, ``p_neg``, ``p_neu``,
    ``label`` (arg-max class name), ``n_tokens``, ``is_boilerplate``.
    """
    import torch

    if not sentences:
        return pd.DataFrame(
            columns=["sentence", "p_pos", "p_neg", "p_neu", "label", "n_tokens", "is_boilerplate"]
        )

    tokenizer, model, label_index, device = model_bundle or load_model(cfg)

    probabilities: list[np.ndarray] = []
    token_counts: list[int] = []

    for start in range(0, len(sentences), cfg.sentiment.batch_size):
        batch = sentences[start : start + cfg.sentiment.batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.sentiment.max_length,
        )
        # Count real (non-padding) tokens so truncation is observable.
        token_counts.extend(encoded["attention_mask"].sum(dim=1).tolist())
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
        probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())

    probs = np.vstack(probabilities)

    # ⚠️ By NAME, never by position. See the module docstring.
    p_pos = probs[:, label_index["positive"]]
    p_neg = probs[:, label_index["negative"]]
    p_neu = probs[:, label_index["neutral"]]

    index_to_label = {v: k for k, v in label_index.items()}
    labels = [index_to_label[i] for i in probs.argmax(axis=1)]

    return pd.DataFrame(
        {
            "sentence": sentences,
            "p_pos": p_pos,
            "p_neg": p_neg,
            "p_neu": p_neu,
            "label": labels,
            "n_tokens": token_counts,
            "is_boilerplate": [is_boilerplate(s) for s in sentences],
        }
    )


def aggregate(sentence_scores: pd.DataFrame) -> dict:
    """Collapse sentence scores to one document score, both formulas.

    Returns ``S_count``, ``S_prob``, ``n_sentences``, ``n_pos``, ``n_neg``,
    ``n_neu``, ``n_boilerplate``, ``mean_confidence``, ``max_tokens``.
    """
    n = len(sentence_scores)
    if n == 0:
        return {
            "S_count": np.nan,
            "S_prob": np.nan,
            "n_sentences": 0,
            "n_pos": 0,
            "n_neg": 0,
            "n_neu": 0,
            "n_boilerplate": 0,
            "mean_confidence": np.nan,
            "max_tokens": 0,
        }

    counts = sentence_scores["label"].value_counts()
    n_pos = int(counts.get("positive", 0))
    n_neg = int(counts.get("negative", 0))

    return {
        "S_count": (n_pos - n_neg) / n,
        "S_prob": float((sentence_scores["p_pos"] - sentence_scores["p_neg"]).mean()),
        "n_sentences": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_neu": int(counts.get("neutral", 0)),
        "n_boilerplate": int(sentence_scores["is_boilerplate"].sum()),
        # Confidence = the arg-max probability. Low mean confidence would
        # suggest FinBERT finds Fed prose genuinely ambiguous, which is itself
        # a finding about domain transfer.
        "mean_confidence": float(sentence_scores[["p_pos", "p_neg", "p_neu"]].max(axis=1).mean()),
        "max_tokens": int(sentence_scores["n_tokens"].max()),
    }


def run_corpus(
    cfg: Config, documents: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the whole corpus once and write both caches.

    Writes ``data/processed/sentiment_scores.csv`` (one row per document) and
    ``data/processed/sentence_scores.parquet`` (one row per sentence).

    The sentence-level cache exists so the section 8.3 Fed-speak audit can pull
    the most extreme individual sentences without re-running the model, and so
    the boilerplate robustness check is a filter rather than a second pass.
    """
    if documents is None:
        documents = pd.read_parquet(cfg.paths.resolve("data_interim") / "documents.parquet")

    docs = documents[
        (documents["doc_type"] == "statement")
        & (~documents["is_flagged"])
        & (documents["doc_date"] >= cfg.data.start_date)
    ].sort_values("doc_date")

    bundle = load_model(cfg)

    doc_rows: list[dict] = []
    sentence_frames: list[pd.DataFrame] = []

    for _, row in docs.iterrows():
        doc_id = row["doc_date"].strftime("%Y%m%d")
        sentences = split_sentences(row["text"])
        if cfg.sentiment.strip_boilerplate:
            sentences = strip_boilerplate(sentences)

        scores = score_sentences(sentences, cfg, model_bundle=bundle)
        scores.insert(0, "sentence_idx", range(len(scores)))
        scores.insert(0, "doc_date", row["doc_date"])
        scores.insert(0, "doc_id", doc_id)
        sentence_frames.append(scores)

        doc_rows.append(
            {
                "doc_id": doc_id,
                "doc_date": row["doc_date"],
                "is_scheduled": row["is_scheduled"],
                "n_chars": row["n_chars"],
                **aggregate(scores),
            }
        )

    doc_scores = pd.DataFrame(doc_rows)
    sentence_scores = pd.concat(sentence_frames, ignore_index=True)

    processed = cfg.paths.resolve("data_processed")
    processed.mkdir(parents=True, exist_ok=True)
    doc_scores.to_csv(processed / "sentiment_scores.csv", index=False)
    sentence_scores.to_parquet(processed / "sentence_scores.parquet")

    return doc_scores, sentence_scores


def load_sentiment_scores(cfg: Config) -> pd.DataFrame:
    """Load the cached document scores with correct dtypes.

    ⚠️ Always use this rather than a bare ``pd.read_csv``. ``doc_id`` looks like
    ``"20260128"``, so CSV inference parses it as **int64** -- while the parquet
    sentence cache keeps it as **str**. Merging the two on ``doc_id`` would then
    match zero rows and return an EMPTY frame without raising anything: pandas
    considers ``20260128`` and ``"20260128"`` simply unequal.

    That is the exact failure mode this project refuses to tolerate -- a silent
    wrong answer rather than a loud error. Parquet is self-describing and does
    not have this problem; CSV is used here only because the document-level
    cache is small and worth being human-readable.
    """
    path = cfg.paths.resolve("data_processed") / "sentiment_scores.csv"
    return pd.read_csv(path, dtype={"doc_id": str}, parse_dates=["doc_date"])


def load_sentence_scores(cfg: Config) -> pd.DataFrame:
    """Load the cached sentence-level scores (parquet: dtypes are preserved)."""
    return pd.read_parquet(cfg.paths.resolve("data_processed") / "sentence_scores.parquet")
