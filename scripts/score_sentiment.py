"""Phase 2 -- run FinBERT over the corpus and cache the scores.

Run with:  uv run python scripts/score_sentiment.py
           uv run python scripts/score_sentiment.py --boilerplate-report

Writes `sentiment_scores.csv` (per document) and `sentence_scores.parquet`
(per sentence). This is the only expensive step; everything downstream reads
the caches, so Phases 3-4 iterate instantly.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from src.config import load_config
from src.sentiment import resolve_device, run_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boilerplate-report",
        action="store_true",
        help="quantify how much of the corpus is vote-tally boilerplate",
    )
    args = parser.parse_args()

    cfg = load_config()
    pd.set_option("display.width", 200)

    device = resolve_device(cfg)
    print(f"Model            : {cfg.sentiment.model}")
    print(f"Device           : {device}")
    print(f"Batch size       : {cfg.sentiment.batch_size}")
    print(f"Max tokens       : {cfg.sentiment.max_length}")
    print(f"Aggregation      : {cfg.sentiment.aggregation}  (PRIMARY, pre-registered)")
    print(f"Strip boilerplate: {cfg.sentiment.strip_boilerplate}  (PRIMARY)")
    print()

    started = time.perf_counter()
    docs, sentences = run_corpus(cfg)
    elapsed = time.perf_counter() - started

    print(f"Scored {len(docs)} documents / {len(sentences)} sentences in {elapsed:.1f}s")
    print(f"  ({len(sentences) / elapsed:.0f} sentences/sec on {device})")

    print("\n--- Document score distribution ---")
    print(
        docs[["S_prob", "S_count", "n_sentences", "mean_confidence"]]
        .describe()
        .round(4)
        .to_string()
    )

    print("\n--- Sanity invariants ---")
    prob_sum = (sentences[["p_pos", "p_neg", "p_neu"]].sum(axis=1) - 1.0).abs().max()
    print(f"  max |p_pos+p_neg+p_neu - 1|      : {prob_sum:.2e}   (softmax invariant)")
    print(f"  S_prob within [-1, 1]            : {bool(docs['S_prob'].between(-1, 1).all())}")
    print(f"  S_count within [-1, 1]           : {bool(docs['S_count'].between(-1, 1).all())}")
    print(f"  documents with zero sentences    : {int((docs['n_sentences'] == 0).sum())}")
    print(
        f"  longest sentence (tokens)        : {int(sentences['n_tokens'].max())} "
        f"/ {cfg.sentiment.max_length} limit"
    )
    n_truncated = int((sentences["n_tokens"] >= cfg.sentiment.max_length).sum())
    print(
        f"  sentences hitting the limit      : {n_truncated}"
        f"{'  <- text was DISCARDED by truncation' if n_truncated else ''}"
    )

    print("\n--- Label mix across the whole corpus ---")
    mix = sentences["label"].value_counts()
    for label, n in mix.items():
        print(f"  {label:9s}: {n:6d}  ({n / len(sentences) * 100:5.1f}%)")

    print("\n--- Correlation between the two aggregations ---")
    pearson = docs["S_prob"].corr(docs["S_count"])
    spearman = docs["S_prob"].corr(docs["S_count"], method="spearman")
    print(f"  Pearson  : {pearson:.4f}")
    print(f"  Spearman : {spearman:.4f}")
    print("  (High correlation => the pre-registered choice barely matters,")
    print("   which is itself worth reporting. Low => they measure different things.)")

    print("\n--- Sentiment by era (is the signal stationary?) ---")
    era = docs.assign(year=docs["doc_date"].map(lambda d: d.year))
    bucket = pd.cut(
        era["year"],
        bins=[1999, 2007, 2015, 2021, 2027],
        labels=["2000-07 pre-GFC", "2008-15 ZIRP", "2016-21", "2022-26 hiking"],
    )
    print(
        era.groupby(bucket, observed=True)["S_prob"]
        .agg(["count", "mean", "std"])
        .round(4)
        .to_string()
    )

    if args.boilerplate_report:
        print("\n--- Boilerplate contamination ---")
        n_bp = int(sentences["is_boilerplate"].sum())
        print(
            f"  boilerplate sentences : {n_bp} / {len(sentences)} "
            f"({n_bp / len(sentences) * 100:.1f}%)"
        )
        print("\n  How FinBERT scores boilerplate (it SHOULD be neutral):")
        bp = sentences[sentences["is_boilerplate"]]
        if not bp.empty:
            print(bp["label"].value_counts().to_string())
            print(
                f"\n  mean (p_pos - p_neg) on boilerplate : "
                f"{(bp['p_pos'] - bp['p_neg']).mean():+.4f}"
            )
            print(
                f"  mean (p_pos - p_neg) on real text   : "
                f"{(sentences[~sentences['is_boilerplate']].eval('p_pos - p_neg')).mean():+.4f}"
            )
            print("\n  Sample boilerplate sentences and their labels:")
            for _, r in bp.head(5).iterrows():
                print(f"    [{r['label']:8s}] {r['sentence'][:96]}")

    print("\n--- Most extreme sentences (the section 8.3 sanity check) ---")
    real = sentences[~sentences["is_boilerplate"]].copy()
    real["net"] = real["p_pos"] - real["p_neg"]
    print("\n  TOP 5 MOST POSITIVE (dovish-flavoured?):")
    for _, r in real.nlargest(5, "net").iterrows():
        print(f"    {r['net']:+.3f}  [{r['doc_id']}]  {r['sentence'][:92]}")
    print("\n  TOP 5 MOST NEGATIVE (hawkish-flavoured?):")
    for _, r in real.nsmallest(5, "net").iterrows():
        print(f"    {r['net']:+.3f}  [{r['doc_id']}]  {r['sentence'][:92]}")

    processed = cfg.paths.resolve("data_processed")
    print(f"\nWrote {processed / 'sentiment_scores.csv'}")
    print(f"Wrote {processed / 'sentence_scores.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
