# STATUS — where the project stands right now

> **Last updated:** 2026-08-04 · **Current phase:** 1 (not started)
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
| **0** | Scaffolding, environment, typed config, test harness | ✅ **complete** 2026-08-04 | 31 tests pass; GPU kernel verified |
| **1** | FOMC statements + SPY prices + leak-free alignment → `panel.parquet` | ⬜ **next** | — |
| **2** | FinBERT scoring → `sentiment_scores.csv` | ⬜ blocked on 1 | — |
| **3** | Trailing Z-score signal + positions | ⬜ blocked on 2 | — |
| **4** | IC, bootstrap CIs, diagnostics, research notebook | ⬜ blocked on 3 | — |
| **5** | Robustness sweeps + minutes corpus (stretch) | ⬜ | — |

**Rule:** do not start phase *N+1* until phase *N*'s Definition of Done in [PLAN.md §7](PLAN.md#7-phase-by-phase-execution-plan) is green.

---

## 2. What is done

### Phase 0 — complete, 2026-08-04

**Environment**
- `.venv` on **Python 3.12.12**, managed by `uv` 0.9.7.
- `pyproject.toml` + `uv.lock` → full-graph reproducible install. `requirements.txt` (536 lines) exported for conventional readers.
- **`torch 2.11.0+cu128`** pinned to the CUDA 12.8 index. Verified `sm_120` cubins present and **a real matmul kernel launches and returns a numerically correct result** — not merely `cuda.is_available()`, which is the check that lies. See [PLAN.md §6.2.2](PLAN.md#622-cuda-on-blackwell--the-trap-that-passes-every-preliminary-check).
- `transformers 5.14.1`; FinBERT loads and runs on GPU.

**Code**
- `config.yaml` — the pre-registered primary configuration, heavily commented.
- `src/config.py` — pydantic schema; `extra="forbid"`, `frozen=True`, cross-field validators.
- `src/{scrape_fomc,prices,align,sentiment,alpha_signal,backtest,diagnostics}.py` — full interfaces, docstrings and contracts written; bodies raise `NotImplementedError` with the phase number.
- `scripts/check_gpu.py`, `conftest.py`, `.gitattributes`, `README.md`.

**Tests — 31 passed, 15 skipped, ruff clean**
- `test_environment.py` (21) — 18 dependency imports, Python version, torch arch list, real GPU matmul.
- `test_config.py` (10) — parses; **primary values pinned**; typo'd keys raise; `auto_adjust`/`Open` consistency; emergency-meeting overrides survive YAML round-tripping.
- 15 skipped canaries for Phases 1–3, written in advance so those phases are built *against* their invariants.

### Documentation — 2026-08-04

- **PLAN.md reconciled with reality.** Two errors corrected, §§3.5/6.1/6.2/6.3/7/10/11 and Appendix A updated, four new sections added (§13 component reference, §14 methodology deep-dive, §15 data dictionary, §16 revision log) plus the §0.2 maintenance protocol.
- **STATUS.md** (this file) created.
- **CLAUDE.md** created — working agreements so the maintenance protocol survives across sessions.

---

## 3. Findings so far

Things learned that were not in the plan. Each changed something downstream.

| # | Finding | Consequence |
|---|---|---|
| **F1** | **FinBERT's `id2label` is `{0: positive, 1: negative, 2: neutral}`** — not alphabetical, not the common convention. | Indexing logits by position **silently inverts the entire signal** with nothing crashing. Must index by name. Pinned by a test. [§2.3](PLAN.md#23--finbert) |
| **F2** | FinBERT scores *"The Committee remains highly attentive to inflation risks"* as **34% positive**, 64% neutral, 2% negative. | A hawkish sentence read as positive sentiment — a live instance of the §2.7 sign trap, on day one. Raises the prior that §8.3's audit will find real disagreement. |
| **F3** | `Adj Close` **cannot** yield an adjusted Open; the plan's §4.3 and Appendix A contradicted each other. | Switched to `auto_adjust: true` (adjusts all four OHLC columns). Averted a ~1.3%/yr dividend bias. [§3.4](PLAN.md#34-price-data-spy) |
| **F4** | `release_date + 1` breaks on **2020-03-15** (Sunday 17:00 ET) and **2008-10-08** (07:00 ET, pre-open). | `release_datetime` is tz-aware; entry via `searchsorted(side="right")` into the real trading calendar. [§3.3](PLAN.md#33-timestamp-alignment-rule-the-heart-of-leak-prevention) |
| **F5** | Fed statement URLs changed format **three times**; index pages split at 2021. | Scraper crawls **index pages** and harvests anchors — never constructs document URLs. Strictly more robust and less code. [Phase 1](PLAN.md#7-phase-by-phase-execution-plan) |
| **F6** | Feb 3–4 1994 is filed under `19940204` — **day 2**. | Confirms the `event_date` = decision-day convention at the very start of the sample. |
| **F7** | **Power analysis: MDE ≈ 0.12 at n ≈ 263, vs. an institution-grade IC of 0.05.** | The study is underpowered by ~an order of magnitude, and we know it *before* running. Reframes the deliverable: the informative output is the power/breadth argument, not the IC. [§14.5](PLAN.md#145--power-analysis--the-most-important-number-in-the-project) |
| **F8** | `transformers` resolved to **5.x**, a major version ahead of the planned 4.x. | 4.x tutorial code should be treated as a hypothesis, not a recipe. Lockfile freezes the version. |

---

## 4. ▶️ Immediate next objectives — Phase 1

**Goal:** a leak-free, timestamped dataset of FOMC text + SPY prices. **No sentiment.**
**Deliverable:** `data/processed/panel.parquet` + `test_align_no_lookahead` passing (all 5 assertions).

Ordered, with the reasoning for the order:

- [ ] **1. `prices.py`** — *first, because it is smaller, independently verifiable, and produces the trading-day index that `align.py` needs as input.*
  - `download_prices()` with `auto_adjust=True`, 1993 → today, cached to `spy_prices.parquet`.
  - `trading_days()` — the sorted session index; **the source of truth**.
  - `validate_against_nyse()` — cross-check both directions; must return an empty frame.

- [ ] **2. `scrape_fomc.py` — discovery**
  - Crawl `fomccalendars.htm` (2021→now) and `fomchistorical{1994..2020}.htm`.
  - Harvest anchors; **do not construct document URLs** (F5).
  - Output the crawl plan for inspection *before* fetching ~270 documents.

- [ ] **3. `scrape_fomc.py` — fetch + manifest**
  - Polite: descriptive `User-Agent`, 1 s delay, exponential backoff, skip URLs already in the manifest.
  - Write raw bytes to `data/raw/fomc_statements/`; log `url, status, bytes, sha256, fetched_at_utc`.
  - ~5 minutes, once, forever.

- [ ] **4. `parse_statement()` + `build_documents_table()`** — *the highest-risk function in the phase.*
  - Defensive selectors across three template eras, with a whole-page text fallback.
  - ⚠️ **Flag short extractions; never pass them through silently.** An empty statement scores as neutral and dilutes the signal without ever failing.
  - Derive `release_datetime` (14:00 ET default + the two overrides) and `release_time_source`.

- [ ] **5. `align.py`** — `next_trading_open()`, `forward_returns()`, `check_overlap()`, `build_panel()`.

- [ ] **6. Un-skip `tests/test_align_no_lookahead.py`** and make all 5 assertions pass.

- [ ] **7. Close the phase** — verify the DoD below, update this file and [PLAN.md §16](PLAN.md#16-revision-log), then **hand Aryan the `git add` / `git commit` commands** (see [CLAUDE.md](CLAUDE.md) — he runs all git himself).

### Phase 1 Definition of Done

- [ ] `validate_against_nyse()` returns an empty discrepancy frame.
- [ ] ~270 statements in `data/raw/`; `verify_manifest()` reports zero hash mismatches.
- [ ] **The realised document count is printed and recorded** — not assumed (F7 depends on knowing the real `n`).
- [ ] Spot-check 5 known meetings against reality: `2001-09-17`, `2008-10-08`, `2008-12-16`, `2020-03-15`, and one recent scheduled meeting.
- [ ] Zero silent drops: `n_scraped == n_parsed + n_flagged`, with the flagged list printed.
- [ ] `test_align_no_lookahead` passes — all 5 assertions.
- [ ] `panel.parquet` has no forward-return NaNs except the unavoidable tail.
- [ ] `check_overlap()` confirms minimum event spacing > 20 trading days, or enumerates the exceptions.

---

## 5. Open questions and accepted risks

| | Item | Decision needed by |
|---|---|---|
| **Q1** | How many statements does the archive actually yield, and how many fail to parse? Drives every t-statistic. | Phase 1 |
| **Q2** | Do the 1994–2005 `/fomc/{date}default.htm` pages contain a *statement*, or only meeting materials? The earliest years may be thinner than the calendar suggests. | Phase 1 |
| **Q3** | Is `quantstats` worth its version risk, or do the hand-rolled metrics suffice? | Phase 4 |
| **Q4** | Given F7, should the project add breadth (rate futures, sector ETFs, intraday) as a Phase 5 objective rather than treating the null as the endpoint? | Phase 4 |
| **R1** | `yfinance` is an unofficial scraper — it can silently return partial history. Mitigated by the parquet cache and the NYSE cross-check, not eliminated. | accepted |
| **R2** | Regime dependence: the Fed↔market relationship differs between ZIRP, tightening and crisis periods. Discussed, not modelled. | accepted until Phase 5 |

---

## 6. How to pick up from here

```bash
uv sync                                  # restore the exact environment
uv run pytest                            # expect 31 passed, 15 skipped
uv run python scripts/check_gpu.py       # expect PASS on the kernel launch
```

Then open [PLAN.md §7, Phase 1](PLAN.md#7-phase-by-phase-execution-plan) and start with `src/prices.py`.
