# STATUS — where the project stands right now

> **Last updated:** 2026-08-04 · **Current phase:** 2 (not started)
>
> This file is the **current position**. [PLAN.md](PLAN.md) is the **complete picture** — theory,
> methodology, architecture, every phase. Read STATUS.md when you sit down to work;
> read PLAN.md when you need to understand or explain anything.
>
> Per [PLAN.md §0.2](PLAN.md#02--maintenance-protocol-a-standing-rule), this file is rewritten
> whenever work happens. It is a snapshot, not a history — the history lives in
> [PLAN.md §16](PLAN.md#16-revision-log) and in the git log.

---

## 1. Phase board

| Phase | Scope | State | Evidence |
|---|---|---|---|
| **0** | Scaffolding, environment, typed config, test harness | ✅ **complete** 2026-08-04 | 31 tests; GPU kernel verified |
| **1** | FOMC statements + SPY prices + leak-free alignment → `panel.parquet` | ✅ **complete** 2026-08-04 | **n = 225**; 51 tests; leak guard PASS |
| **2** | FinBERT scoring → `sentiment_scores.csv` | ⬜ **next** | — |
| **3** | Trailing Z-score signal + positions | ⬜ blocked on 2 | — |
| **4** | IC, bootstrap CIs, diagnostics, research notebook | ⬜ blocked on 3 | — |
| **5** | Robustness sweeps + minutes corpus (149 docs already scraped) | ⬜ | — |

**Rule:** do not start phase *N+1* until phase *N*'s Definition of Done in [PLAN.md §7](PLAN.md#7-phase-by-phase-execution-plan) is green.

---

## 2. The dataset, in one table

| | |
|---|---|
| **Primary sample** | **225 statements**, 2000-02-02 → 2026-07-29 |
| Composition | 212 scheduled · 13 unscheduled (tagged, kept) |
| Usable for the IC | **219** (after the *L*=6 Z-score warm-up) |
| **Minimum detectable IC** | **≈ 0.13** vs. an institution-grade signal of 0.05 |
| Prices | SPY, 8,434 sessions, 1993-01-29 → 2026-08-03, auto-adjusted |
| NYSE cross-check | **zero discrepancies** across 33.5 years |
| Raw corpus | 393 documents (244 statements + 149 minutes), sha256-verified |
| Also stored | 19 pre-2000 statements, labelled, for the regime robustness check |

---

## 3. What is done

### Phase 0 — 2026-08-04
Python 3.12 + `uv` + `uv.lock`; `torch 2.11.0+cu128` verified by a **real kernel launch** on the RTX 5070 Ti (`sm_120`); pydantic-validated `config.yaml` with pre-registered values pinned by a test; leak canaries written in advance.

### Phase 1 — 2026-08-04

**`src/prices.py`** — SPY via yfinance (`auto_adjust=True`), cached to parquet with a provenance sidecar. `validate_against_nyse()` cross-checks the price index against the official NYSE session list **in both directions** and returns empty.

**`src/scrape_fomc.py`** — crawls index pages and harvests anchors; **never constructs document URLs**. Polite (1 s delay, backoff, manifest-based skip). Every fetch logged with URL, status, bytes, sha256, UTC time. Parses defensively across four template eras, extracts each document's **self-stated release time**, and flags both short parses and soft-404 bodies.

**`src/align.py`** — the leak guard. Entry = first session whose **09:30 open** is strictly after `release_datetime`, via `searchsorted(side="right")`. Vectorised forward returns at *h* ∈ {1,3,5,10,20}, in sessions. `check_overlap()` enumerates every overlapping pair. `build_panel()` **asserts** no look-ahead at build time, not just in tests.

**Tests — 51 passed, 10 skipped, ruff clean.** `test_align_no_lookahead.py` is 20 tests in two layers: 11 synthetic (hand-built calendar with fabricated holidays, Sunday and pre-open releases, naive-timestamp rejection) and 9 integration against the real panel.

---

## 4. Findings so far

| # | Finding | Consequence |
|---|---|---|
| **F1** | FinBERT's `id2label` is `{0: positive, 1: negative, 2: neutral}` | Positional indexing **silently inverts the signal**. Index by name. |
| **F2** | FinBERT scores a hawkish inflation sentence as 34% **positive** | Live instance of the §2.7 sign trap, day one. |
| **F3** | `Adj Close` cannot yield an adjusted Open | `auto_adjust: true`; averted a ~1.3%/yr dividend bias. |
| **F4** | `release_date + 1` breaks on Sunday and pre-open releases | tz-aware timestamps + `searchsorted` into the real calendar. |
| **F5** | Statement URLs changed format **4 times** (not 3) | The `boarddocs` era, **61 statements / 9 years**, was missing from the plan. Index-crawling vindicated. |
| **F6** | Feb 3–4 1994 filed under `19940204` | `doc_date` = **publication** date, not meeting date. |
| **F7** | **MDE ≈ 0.13 at n = 219** vs. 0.05 for a strong signal | Underpowered by ~an order of magnitude, known *before* running. |
| **F8** | `transformers` resolved to 5.x | Treat 4.x tutorial code as a hypothesis. |
| **F9** | 🔧 **Disclosure regime changed**: Fed announced only policy *changes* until 1999; **1996 has zero statements** | Selection effect + a 28-month Z-window posing as 9 months. **Sample start 1994 → 2000**, costing 8% of data. [§3.1.1](PLAN.md#311--the-disclosure-regime-changed--measured-in-phase-1) |
| **F10** | 🔧 **Documents state their own release time** (`"For release at 2:00 p.m. EDT"`) | 88/225 timestamps are now evidence, not convention. Both hand overrides independently **confirmed**. [§3.3.1](PLAN.md#331--refinement-forced-by-phase-1-compare-against-the-session-open-not-the-date) |
| **F11** | 🔧 **Overlap was materially understated.** Min gap **3 sessions**, not ~32; **16 overlapping pairs at *h*=20** | The plan's "overlap is minimal" used the *average* and ignored the *minimum*. Emergency meetings cluster inside gaps, during crises. Headline → short horizons; Newey–West at *h* ≥ 10. [§4.3](PLAN.md#43-step-c--forward-returns-r_tto-th) |
| **F12** | 🔧 A Fed **soft-404 body is ~1,170 chars** — above the 500-char flag threshold | Length checks alone are insufficient; content is checked too. |

---

## 5. ▶️ Immediate next objectives — Phase 2 (FinBERT)

**Goal:** turn each of the 225 statements into a raw sentiment score, cached so Phases 3–4 iterate instantly.
**Deliverable:** `sentiment_scores.csv` (one row per document) + `sentence_scores.parquet` (one row per sentence).

- [ ] **1. `resolve_device()` + model loading** — `ProsusAI/finbert`, `model.eval()`, `torch.no_grad()`, fixed seed. ⚠️ Read `id2label` at load time and index by **name** (F1).
- [ ] **2. `split_sentences()`** — `nltk.sent_tokenize` with the regex fallback, so the pipeline never hard-depends on a runtime `punkt` download.
- [ ] **3. `score_sentences()`** — batched under `no_grad()`, `truncation=True, max_length=512`.
- [ ] **4. `aggregate()`** — both `S_count` and `S_prob` (PLAN.md §4.1). Primary is pre-registered as `prob`.
- [ ] **5. Run the corpus once**, write both caches. ~15–20k sentences; seconds on the GPU.
- [ ] **6. Un-skip `tests/test_sentiment_shapes.py`** — 6 tests, including the `id2label` pin and determinism.
- [ ] **7. Close the phase** — update PLAN.md §16 + this file, then hand over the `git add` / `git commit` commands.

### Phase 2 Definition of Done
- [ ] `sentiment_scores.csv` has exactly 225 rows (or the discrepancy is explained), both scores populated.
- [ ] Sentence-level cache saved (needed by §8.3 without re-running FinBERT).
- [ ] Probabilities sum to 1; scores within [−1, 1]; no document lost.
- [ ] `id2label` assertion passes.
- [ ] **Re-running produces byte-identical output.**
- [ ] Boilerplate contamination quantified: what fraction of sentences are vote tallies?

---

## 6. Open questions and accepted risks

| | Item | Decision needed by |
|---|---|---|
| **Q3** | Is `quantstats` worth its version risk, or do the hand-rolled metrics suffice? | Phase 4 |
| **Q4** | Given F7, should Phase 5 pursue **breadth** (rate futures, sector ETFs, intraday) rather than treating a null as the endpoint? | Phase 4 |
| **Q5** | 🔧 Given F11, should the **primary horizon move from *h*=5 to *h*=3**? *h*=3 is provably overlap-free; *h*=5 has one overlapping pair. ⚠️ This is a **pre-registered primary value** — changing it needs a reason recorded *before* seeing any IC. | **Before Phase 4** |
| **Q6** | 🔧 63 of 244 statements parsed via `whole_page_fallback` (the 1997–2005 `boarddocs` era). Text lengths look sane, but should era-specific selectors be added to reduce nav-boilerplate contamination? | Phase 2 |
| **R1** | `yfinance` is unofficial — mitigated by the parquet cache and NYSE cross-check, not eliminated. | accepted |
| **R2** | Regime dependence (ZIRP vs. tightening vs. crisis). Discussed, not modelled. | accepted until Phase 5 |
| **R3** | 🔧 137 of 225 release times are the 14:00 **default**, not verified. Conservative by construction (erring late costs signal; erring early would leak), but they are assumptions. | accepted, documented |

**Resolved:** ~~Q1~~ (n = 225, measured) · ~~Q2~~ (the 1994–2005 pages do contain real statements; the sparse years are a disclosure-regime artefact, not a scraping failure — F9).

---

## 7. How to pick up from here

```bash
uv sync
uv run pytest                                    # expect 51 passed, 10 skipped
uv run python scripts/scrape_fomc.py --verify    # expect manifest PASS
uv run python scripts/build_panel.py             # rebuild + print the full report
```

Then open [PLAN.md §7, Phase 2](PLAN.md#7-phase-by-phase-execution-plan) and start with `src/sentiment.py`.
