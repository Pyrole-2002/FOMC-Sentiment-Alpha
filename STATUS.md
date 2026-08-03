# STATUS — where the project stands right now

> **Last updated:** 2026-08-04 · **Current phase:** 3 (not started)
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
| **0** | Scaffolding, environment, typed config, test harness | ✅ **complete** | 31 tests; GPU kernel verified |
| **1** | FOMC statements + SPY prices + leak-free alignment → `panel.parquet` | ✅ **complete** | **n = 225**; leak guard PASS |
| **2** | FinBERT scoring → `sentiment_scores.csv` | ✅ **complete** | 3,846 sentences; **byte-identical re-runs** |
| **3** | Trailing Z-score signal + positions | ⬜ **next** | — |
| **4** | IC, bootstrap CIs, diagnostics, research notebook | ⬜ blocked on 3 | — |
| **5** | Robustness sweeps + minutes corpus (149 docs already scraped) | ⬜ | — |

**72 tests passing, 4 skipped** (the remaining skips are Phase 3 canaries). Ruff clean.

**Rule:** do not start phase *N+1* until phase *N*'s Definition of Done in [PLAN.md §7](PLAN.md#7-phase-by-phase-execution-plan) is green.

---

## 2. The dataset, in one table

| | |
|---|---|
| **Primary sample** | **225 statements**, 2000-02-02 → 2026-07-29 |
| Composition | 212 scheduled · 13 unscheduled (tagged, kept) |
| Usable for the IC | **219** (after the *L*=6 Z-score warm-up) |
| **Minimum detectable IC** | **≈ 0.13** vs. an institution-grade signal of 0.05 |
| Sentences scored | **3,846** (17.1 per document, longest 177 tokens vs the 512 limit) |
| Label mix | 49.3% neutral · 31.0% positive · 19.7% negative |
| Boilerplate | **11.1%** of sentences, mean net −0.065 vs +0.124 on real text |
| `S_prob` vs `S_count` | ρ = **0.971** (Pearson), 0.967 (Spearman) |
| Prices | SPY, 8,434 sessions, auto-adjusted, **zero NYSE discrepancies** |
| Raw corpus | 393 documents, sha256-verified |

---

## 3. What is done

**Phase 0** — Python 3.12 + `uv` + lockfile; `torch 2.11.0+cu128` verified by a real kernel launch on the RTX 5070 Ti (`sm_120`); pydantic-validated config with pre-registered values pinned by a test.

**Phase 1** — `prices.py` (SPY + NYSE cross-check), `scrape_fomc.py` (index-crawling, sha256 provenance, four template eras, self-stated release times), `align.py` (session-open entry rule via `searchsorted`, vectorised forward returns, overlap enumeration, build-time leak assertion).

**Phase 2** — `sentiment.py`: FinBERT on GPU, label lookup **by name**, batched under `no_grad()`, both aggregations, boilerplate detection, nltk-with-regex-fallback splitting. Two caches written; `load_sentiment_scores()` is the sanctioned reader.

---

## 4. Findings so far

| # | Finding | Consequence |
|---|---|---|
| **F1** | FinBERT's `id2label` is `{0: positive, 1: negative, 2: neutral}` | Positional indexing **silently inverts the signal**. Pinned by a test. |
| **F3** | `Adj Close` cannot yield an adjusted Open | `auto_adjust: true`; averted a ~1.3%/yr dividend bias. |
| **F4** | `release_date + 1` breaks on Sunday and pre-open releases | tz-aware timestamps + `searchsorted` into the real calendar. |
| **F5** | Statement URLs changed format **4 times** (not 3) | The `boarddocs` era, 61 statements / 9 years, was missing from the plan. |
| **F7** | **MDE ≈ 0.13 at n = 219** vs 0.05 for a strong signal | Underpowered by ~an order of magnitude, known *before* running. |
| **F9** | **Disclosure regime changed**: only policy *changes* announced until 1999; **1996 has zero statements** | Sample start 1994 → 2000. Selection effect + a 28-month Z-window posing as 9 months. |
| **F10** | **Documents state their own release time** | 88/225 timestamps are evidence, not convention. Both hand overrides **confirmed**. |
| **F11** | **Overlap was understated.** Min gap **3 sessions**; 16 overlapping pairs at *h*=20 | Headline stays at short horizons; Newey–West at *h* ≥ 10. |
| **F12** | A Fed **soft-404 body is ~1,170 chars** — above the flag threshold | Length checks alone are insufficient; content is checked too. |
| **F13** | 🔧 **THE SIGN TRAP IS CONFIRMED.** FinBERT's extreme sentences are *all* descriptions of economic conditions — **not one** describes policy stance | FinBERT answers *"is the economy good?"*, which for equities is near the **inverse** of *"is the Fed dovish?"*. **A negative IC is now the economically coherent prediction.** [§2.7.1](PLAN.md#271--the-sign-trap-is-confirmed--measured-in-phase-2) |
| **F14** | 🔧 **Tone is strongly non-stationary**: mean `S_prob` +0.168 (pre-GFC) → +0.039 (ZIRP) → +0.133 → +0.055 (hiking) | Swings exceed a within-era sd. The Z-score is **load-bearing**: raw levels would produce spurious correlation with any trending series. |
| **F15** | 🔧 **Boilerplate is 11.1% and tilts negative** (−0.065 vs +0.124), and is **era-correlated** (constant-length vote tallies are a bigger share of shorter early statements) | A time-varying bias can masquerade as a trend. Robustness run with `strip_boilerplate: true`. |
| **F16** | 🔧 **`doc_id` is int64 from CSV, str from parquet** | A Phase 3 merge would have matched **zero rows and returned an empty frame silently**. Fixed with `load_sentiment_scores()`; dtype pinned by a test. |

---

## 5. ▶️ Immediate next objectives — Phase 3 (the alpha signal)

**Goal:** convert raw sentiment into the tradable Z-score and positions.
**Deliverable:** `panel.parquet` augmented with `S`, `mu`, `sigma`, `Z`, `position`; both leak canaries green.

- [ ] **1. `rolling_zscore()`** — trailing mean/sd with the mandatory `.shift(1)`. ⚠️ The `.shift(1)` is the single most important line in the phase: without it $S_t$ enters its own mean.
- [ ] **2. `positions_from_z()`** — threshold rule, sign from the pre-registered hypothesis (`dovish_surprise_bullish`, i.e. +1). ⚠️ **Do not flip it** on seeing F13 — F13 is a *prediction* about what the test will show, not a licence to fit the sign.
- [ ] **3. `build_signal()`** — merge via `load_sentiment_scores()` (F16), not a bare `read_csv`.
- [ ] **4. Un-skip `tests/test_zscore_no_lookahead.py`** — the canary must rebuild every $Z_t$ with an independent backward loop, sharing no code with the implementation.
- [ ] **5. Close the phase** — update PLAN.md §16 + this file, hand over the `git` commands.

### Phase 3 Definition of Done
- [ ] `Z` reproduced exactly by a naive loop over rows `< t`.
- [ ] Perturbing a future $S$ leaves every past $Z$ unchanged.
- [ ] First *L* = 6 rows have `NaN` Z and zero position.
- [ ] `|Z| ≤ θ` ⇒ position 0; sign of position follows sign of Z.
- [ ] Position counts reported (how many long / short / flat) — a rule that never trades is a finding.
- [ ] **No IC computed.** That is Phase 4. Looking early would undermine the pre-registration.

---

## 6. Open questions and accepted risks

| | Item | Decision needed by |
|---|---|---|
| **Q3** | Is `quantstats` worth its version risk, or do the hand-rolled metrics suffice? | Phase 4 |
| **Q4** | Given F7, should Phase 5 pursue **breadth** (rate futures, sector ETFs, intraday) rather than treating a null as the endpoint? | Phase 4 |
| **Q7** | 🔧 Given F13, should §8.3's hand-labelling be **two-axis** (hawk/dove *and* good/bad economy) to measure *which* axis FinBERT tracks? Recommended — it converts F13 from anecdote to measurement. ~30 sentences of your labelling time. | Phase 4 |
| **R1** | `yfinance` is unofficial — mitigated, not eliminated. | accepted |
| **R2** | Regime dependence. Discussed, not modelled. | accepted until Phase 5 |
| **R3** | 137 of 225 release times are the 14:00 **default**, not verified. Conservative by construction. | accepted, documented |

**Resolved:** ~~Q1~~ (n = 225) · ~~Q2~~ (sparse early years are a disclosure-regime artefact — F9) · ~~Q5~~ (**keep *h*=5**; the overlap costs 0.4% of effective sample, while revising a pre-registered spec after inspecting data costs far more credibility — reasoning recorded in `config.yaml`) · ~~Q6~~ (fallback parses are clean; the length gap is real history, not a parser artefact).

---

## 7. How to pick up from here

```bash
uv sync
uv run pytest                                    # expect 72 passed, 4 skipped
uv run python scripts/build_panel.py             # rebuild the panel + full report
uv run python scripts/score_sentiment.py --boilerplate-report
```

Then open [PLAN.md §7, Phase 3](PLAN.md#7-phase-by-phase-execution-plan) and start with `src/alpha_signal.py`.
