# STATUS — where the project stands right now

> **Last updated:** 2026-08-04 · **Current phase:** 4 (not started)
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
| **0** | Scaffolding, environment, typed config, test harness | ✅ **complete** | GPU kernel verified |
| **1** | FOMC statements + SPY prices + leak-free alignment | ✅ **complete** | **n = 225**; entry-leak guard PASS |
| **2** | FinBERT scoring → `sentiment_scores.csv` | ✅ **complete** | 3,846 sentences; byte-identical re-runs |
| **3** | Trailing Z-score signal + positions | ✅ **complete** | **Z-leak canary PASS**; 55L / 50S / 120 flat |
| **4** | IC, bootstrap CIs, diagnostics, research notebook | ⬜ **next** | — |
| **5** | Robustness sweeps + minutes corpus (149 docs already scraped) | ⬜ | — |

**95 tests passing, ZERO skipped.** Ruff clean.

**Rule:** do not start phase *N+1* until phase *N*'s Definition of Done in [PLAN.md §7](PLAN.md#7-phase-by-phase-execution-plan) is green.

---

## 2. The dataset and the signal, in one table

| | |
|---|---|
| **Primary sample** | **225 statements**, 2000-02-02 → 2026-07-29 |
| Composition | 212 scheduled · 13 unscheduled (tagged, kept) |
| **Usable for the IC** | **219** (after the *L*=6 warm-up) |
| **Minimum detectable IC** | **≈ 0.13** vs an institution-grade signal of 0.05 |
| Sentences scored | 3,846 · 49.3% neutral, 31.0% positive, 19.7% negative |
| **Z** mean / sd | +0.165 / **1.790** (⚠️ *not* 1 — see F17) |
| **Positions** | **55 long · 50 short · 120 flat** — in market 46.7% |
| Turnover | 119 total, 0.529 per event |
| Prices | SPY, 8,434 sessions, **zero NYSE discrepancies** |

**Nothing in Phases 1–3 has examined a return column.** Evaluation is Phase 4.

---

## 3. What is done

**Phase 0** — Python 3.12 + `uv` + lockfile; `torch 2.11.0+cu128` verified by a real kernel launch (`sm_120`); pydantic-validated config with pre-registered values pinned by a test.

**Phase 1** — `prices.py`, `scrape_fomc.py` (index-crawling, sha256 provenance, four template eras, self-stated release times), `align.py` (session-open entry via `searchsorted`, vectorised forward returns, overlap enumeration, build-time leak assertion).

**Phase 2** — `sentiment.py`: FinBERT on GPU, label lookup **by name**, both aggregations, boilerplate detection, byte-identical caches.

**Phase 3** — `alpha_signal.py`: trailing Z-score with the mandatory `.shift(1)`, threshold rule, turnover. The leak canary is asserted **both** at build time and in tests, via a naive backward loop that shares no code with the implementation — a test that used `.rolling().shift()` could not detect a missing shift, because both would be wrong identically.

---

## 4. Findings so far

| # | Finding | Consequence |
|---|---|---|
| **F1** | FinBERT's `id2label` is `{0: positive, 1: negative, 2: neutral}` | Positional indexing **silently inverts the signal**. Pinned by a test. |
| **F3** | `Adj Close` cannot yield an adjusted Open | `auto_adjust: true`; averted a ~1.3%/yr dividend bias. |
| **F4** | `release_date + 1` breaks on Sunday and pre-open releases | tz-aware timestamps + `searchsorted` into the real calendar. |
| **F5** | Statement URLs changed format **4 times** (not 3) | The `boarddocs` era, 61 statements / 9 years, was missing from the plan. |
| **F7** | **MDE ≈ 0.13 at n = 219** vs 0.05 for a strong signal | Underpowered by ~an order of magnitude, known *before* running. |
| **F9** | **Disclosure regime changed**; **1996 has zero statements** | Sample start 1994 → 2000. |
| **F10** | **Documents state their own release time** | 88/225 timestamps are evidence, not convention. |
| **F11** | **Overlap understated.** Min gap **3 sessions**; 16 pairs at *h*=20 | Newey–West at *h* ≥ 10. |
| **F12** | A Fed **soft-404 body is ~1,170 chars** | Length checks alone are insufficient. |
| **F13** | **THE SIGN TRAP IS CONFIRMED.** FinBERT's extremes are *all* economic-condition descriptions — **none** describe policy stance | FinBERT answers *"is the economy good?"* ≈ the **inverse** of *"is the Fed dovish?"*. **A negative IC is the coherent prediction.** [§2.7.1](PLAN.md#271--the-sign-trap-is-confirmed--measured-in-phase-2) |
| **F14** | **Tone is strongly non-stationary** across regimes | Z-score is load-bearing, not cosmetic. |
| **F15** | **Boilerplate is 11.1% and tilts negative**, era-correlated | A time-varying bias can masquerade as a trend. |
| **F16** | **`doc_id` int64 vs str; `doc_date` object vs datetime64** | Would have caused an empty merge / a raise. Normalised at the source; dtypes pinned. |
| **F17** | 🔧 **sd(Z) = 1.79, not 1.** A trailing Z with σ̂ from 5 d.o.f. is ***t*-distributed** (theoretical sd 1.39; the excess is fat tails in *S*) | θ=1.0 is **not** "one sigma" — it trades 47.9% of events, not ~32%. Extremes are real surprises (ρ=+0.80 with numerator) amplified by small σ̂ (ρ=+0.21). *L*=12 already in the robustness grid. [§4.2.1](PLAN.md#421--1-standard-deviation-is-the-wrong-intuition--measured-in-phase-3) |
| **F18** | 🔧 **A dtype fix silently turned 3 passing tests into SKIPS** | `panel["doc_date"] == date(...)` matched nothing. **A skip reads as green.** Conditional skips that depend on dtypes are assertions in disguise — now asserted outright. |

---

## 5. ▶️ Immediate next objectives — Phase 4 (the honest evaluation)

**Goal:** measure whether the signal predicts returns, with error bars, and characterise it.
**Deliverable:** IC grid with bootstrap CIs, three diagnostic plots, the research notebook.

⚠️ **This is the phase everything was pre-registered for.** Primary spec is locked: `S_prob`, *L*=6, θ=1.0, *h*=5, sign +1, Spearman. Everything else is a labelled robustness check.

- [ ] **1. `information_coefficient()`** — Spearman (primary) + Pearson (secondary), with *t*-stat and *n* actually used.
- [ ] **2. `bootstrap_ic_ci()`** — 10,000 resamples. ⚠️ Resample **pairs**, never the two series independently (that destroys the association and always centres on zero).
- [ ] **3. `ic_grid()`** — the **whole grid** (5 horizons × 2 aggregations), never the max.
- [ ] **4. Newey–West standard errors** at *h* ≥ 10, where F11 showed material overlap.
- [ ] **5. `strategy_returns()` + `performance_metrics()`** — hand-computed Sharpe/MDD/win-rate, annualised on **8 events/year**, not 252 days (that would inflate Sharpe ~5.6×).
- [ ] **6. Diagnostics** — IC decay with CI error bars; cost-sensitivity curve over `[0,1,2,5,10]` bps; the §8.3 Fed-speak audit.
- [ ] **7. `99_research_report.ipynb`** — the narrative, ending in an honestly-caveated conclusion, positive *or* negative.

### Phase 4 Definition of Done
- [ ] IC reported **with** *t*-stat and bootstrap CI for **every** horizon and both aggregations.
- [ ] The power analysis (F7) stated **alongside** the result, not as an excuse afterwards.
- [ ] Sign hypothesis explicitly **tested**, and the F13 prediction confirmed or refuted.
- [ ] Cost curve shows where the edge dies.
- [ ] Robustness grid reported as a grid: *L*∈{6,12}, θ∈{0.5,1,1.5}, agg∈{prob,count}, unscheduled on/off, sample-start 2000/1994.
- [ ] Conclusion is falsifiable and honestly caveated — **a rigorous null is a success** (§5.5).

---

## 6. Open questions and accepted risks

| | Item | Decision needed by |
|---|---|---|
| **Q3** | Is `quantstats` worth its version risk, or do the hand-rolled metrics suffice? | Phase 4 |
| **Q4** | Given F7, should Phase 5 pursue **breadth** (rate futures, sector ETFs, intraday) rather than treating a null as the endpoint? | Phase 4 |
| **Q7** | Given F13, should §8.3's hand-labelling be **two-axis** (hawk/dove *and* good/bad economy)? Recommended — converts F13 from anecdote to measurement. ~30 sentences of your time. | Phase 4 |
| **R1** | `yfinance` is unofficial — mitigated, not eliminated. | accepted |
| **R2** | Regime dependence. Discussed, not modelled. | accepted until Phase 5 |
| **R3** | 137 of 225 release times are the 14:00 **default**. Conservative by construction. | accepted, documented |
| **R4** | 🔧 Given F17, *Z* is heavy-tailed (±9.9). Spearman IC is rank-based so it is **already robust** to this — which is exactly why it was pre-registered as primary. Pearson IC will be more affected; the gap between them is informative. | accepted, informative |

**Resolved:** ~~Q1~~ (n=225) · ~~Q2~~ (disclosure-regime artefact) · ~~Q5~~ (**keep *h*=5**) · ~~Q6~~ (fallback parses are clean).

---

## 7. How to pick up from here

```bash
uv sync
uv run pytest                                 # expect 95 passed, 0 skipped
uv run python scripts/build_signal.py         # signal report (no returns shown)
```

Then open [PLAN.md §7, Phase 4](PLAN.md#7-phase-by-phase-execution-plan) and start with `src/backtest.py`.
