# FOMC Sentiment Alpha — Project Plan & Teaching Document

> **Project codename:** `fed-sentiment-alpha`
> **One-line thesis:** *Changes* in the sentiment of Federal Reserve (FOMC) communications contain short-horizon, tradable information about US equity returns (SPY). We will extract that sentiment with a finance-tuned NLP model (FinBERT), convert it into a normalized alpha signal, and evaluate it with the same rigor an institutional quant desk would demand.
>
> **Audience for this document:** you (the builder) — and, indirectly, a QRT-style interviewer reading over your shoulder. Every concept is defined the first time it appears. Every design choice carries a *"why."*
>
> **Status:** **Phase 0 complete** (2026-08-04). Phase 1 is next.
> For the live progress ledger and immediate next objectives, see **[STATUS.md](STATUS.md)**.
> This document is the *complete picture*; STATUS.md is the *current position*.

---

## Table of Contents

0. [How to read this document (and how it is maintained)](#0-how-to-read-this-document)
1. [The big picture: what are we actually building?](#1-the-big-picture)
2. [The Quant Lexicon (concepts you must own)](#2-the-quant-lexicon)
3. [The data reality of the FOMC (read before coding)](#3-the-data-reality-of-the-fomc)
4. [The mathematical methodology, derived and explained](#4-the-mathematical-methodology)
5. [Statistical rigor: why most backtests are lies](#5-statistical-rigor)
6. [Repository & environment architecture](#6-repository--environment-architecture)
7. [Phase-by-phase execution plan](#7-phase-by-phase-execution-plan)
8. [The "QRT Touch": research-grade diagnostics](#8-the-qrt-touch)
9. [Interview narrative & talking points](#9-interview-narrative)
10. [Risk register & known limitations](#10-risk-register)
11. [Glossary quick-reference](#11-glossary-quick-reference)
12. [References & further reading](#12-references)
13. [**Component reference — every module, its contract, its dependencies**](#13-component-reference)
14. [**Methodology deep-dive — the maths, derived from first principles**](#14-methodology-deep-dive)
15. [**Data dictionary — the schema of every artifact**](#15-data-dictionary)
16. [**Revision log**](#16-revision-log)

---

## 0. How to read this document

This is both a **plan** and a **textbook**. It is deliberately long because the whole point of the project is to demonstrate *understanding*, not just working code. When you execute, work through it **top to bottom, phase by phase**, and do not skip Section 5 (Statistical Rigor) — that section is what turns "a student's cool project" into "a researcher's project."

Each phase ends with a **Definition of Done (DoD)** checklist and a **Deliverable**. Do not start phase *N+1* until phase *N*'s DoD is green. This ordering is not bureaucracy — it is the single most important defense against the cardinal sin of quant research (look-ahead bias), which we build the pipeline chronologically to prevent.

**Symbols used:**
- 🧠 = a concept being taught
- ⚠️ = a trap that silently corrupts results
- ✅ = a Definition-of-Done item
- 💬 = an interview talking point you can say out loud
- 🔧 = a decision that *changed* from the original plan, with the reason

### 0.1 The two documents, and the difference between them

| | **PLAN.md** (this file) | **[STATUS.md](STATUS.md)** |
|---|---|---|
| Answers | *What is this project, why is it built this way, and what does every part mean?* | *Where exactly are we, and what is the very next thing to do?* |
| Scope | The complete picture: theory, methodology, architecture, all phases | The current position: what is done, what is next, what is blocked |
| Lifetime | Grows and is corrected; nothing is deleted without a revision-log entry | Rewritten freely; it is a snapshot, not a history |
| Read it when | You need to understand or explain *anything* | You sit down to work and need to know where you left off |

Keeping these separate matters. If progress notes are interleaved with the methodology, the methodology becomes unreadable within a week — and this document's entire purpose is to be the thing you can hand to an interviewer.

### 0.2 🔧 Maintenance protocol (a standing rule)

> **Every change to the code, the environment, the data, or the findings must be reflected in this document *in the same session that makes the change*.** A plan that describes a repository that no longer exists is worse than no plan, because it will be trusted.

Concretely, whenever work happens:

1. **Correct the affected lines in place.** Do not append a "but actually..." note further down. If §3.4 says we use `Adj Close` and we no longer do, §3.4 changes.
2. **Mark deviations with 🔧 and give the reason.** A deviation without a reason looks like drift; a deviation with a reason looks like research. The reasons are also interview material.
3. **Update [STATUS.md](STATUS.md):** move the completed item, restate the immediate next objectives, record anything newly discovered or newly blocked.
4. **Add a line to the [Revision log (§16)](#16-revision-log)** with the date and a one-line summary. (Commits are made by hand — see [CLAUDE.md](CLAUDE.md) — so the log records *what changed*, and `git log` records *when it landed*.)
5. **Update the phase's DoD in §7** from a promise into a record of what was actually verified, including the command that verified it.

⚠️ **The trap this protects against.** In a research project the document *is* the deliverable — the code is only evidence for it. Every hour that PLAN.md and the repository disagree is an hour in which you are building on a false model of your own work, and in which anything you tell an interviewer might be wrong. Stale documentation in a rigor-focused project is not untidiness; it is the same category of error as look-ahead bias — believing something that is not true because you never checked.

---

## 1. The big picture

### 1.1 The intuition in plain English

Eight times a year, the Federal Open Market Committee (FOMC) — the committee inside the Federal Reserve that sets US interest-rate policy — publishes text: a **policy statement** on the meeting day, and detailed **minutes** three weeks later. Markets hang on every word. The *tone* of this language ("the Committee remains highly attentive to inflation risks" vs. "risks to employment have increased") moves trillions of dollars.

Our hypothesis: if we can **measure the tone quantitatively** and, crucially, measure **how the tone changed relative to what the market expected**, we may be able to predict the direction of the S&P 500 over the next few days.

### 1.2 The pipeline at a glance

```
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐   ┌───────────────┐
│  RAW TEXT   │   │   NLP ENGINE │   │  RAW SENTIMENT│   │ ALPHA SIGNAL │   │  EVALUATION   │
│ FOMC docs   │──▶│  FinBERT     │──▶│  score  S_t   │──▶│ Z-score Z_t  │──▶│ IC, Sharpe,   │
│ (scraped)   │   │ (per sentence│   │  per document │   │ + trade rule │   │ decay, costs  │
│ + SPY prices│   │  aggregation)│   │  (cached CSV) │   │              │   │ (tearsheet)   │
└─────────────┘   └──────────────┘   └───────────────┘   └──────────────┘   └───────────────┘
   Phase 1            Phase 2             Phase 2            Phase 3           Phase 4
```

### 1.3 What "success" means here

**Success is NOT "the strategy makes money in the backtest."** A backtest that beats the market proves almost nothing (Section 5 explains why). Success is:

1. A **leak-free** pipeline where every signal at time *t* uses only information available at *t*.
2. A **statistically honest** measurement of whether the signal has predictive power (the Information Coefficient and its t-statistic).
3. A **characterization** of the signal: how fast its predictive power decays, how sensitive it is to trading costs, and whether the model's "understanding" of Fed language is qualitatively sane.

If the honest answer turns out to be *"the edge is weak and disappears after costs,"* that is a **completely acceptable and impressive result** — because you will have *proven* it rigorously. Interviewers at firms like QRT are far more impressed by a correct negative result than by a suspicious positive one.

💬 *"My goal wasn't to find a money-printer — it was to measure whether FOMC sentiment carries short-horizon information, and to quantify my own uncertainty about that answer."*

---

## 2. The Quant Lexicon

These are the terms you must be able to define instantly and use correctly. Memorize the definitions; the *reasoning* is what earns respect.

### 2.1 🧠 Look-Ahead Bias (the cardinal sin)

**Definition:** using, in a decision made at time *t*, any piece of information that was not actually *available* until some later time *t + k*.

**Why it is lethal:** it is the single most common reason backtests look brilliant and live trading loses money. The backtest "peeks at the answer." It is insidious because it hides in innocent-looking code: a `.shift()` in the wrong direction, a rolling window that's centered instead of trailing, a data provider that back-fills revised values into historical dates.

**Concrete example for this project:** The FOMC statement is released at **2:00 PM ET** on a meeting day. The market opened at **9:30 AM ET** that same morning. If your backtest "trades on the statement" at that day's *open*, you have used 2 PM information to trade at 9:30 AM — a 4.5-hour time machine. The fix: you may only act at the **2:00 PM release or later** — realistically the **market close (4:00 PM ET)** of the release day, or the **next day's open**. We will enforce the stricter, cleaner rule (**next trading day's open**) throughout.

**Even worse example — the minutes:** FOMC *minutes* are released **~3 weeks after** the meeting they describe. A naive scraper that stamps the minutes with the *meeting date* would let you "trade" three weeks before the text existed. This is a project-killing bug we design against explicitly (Section 3).

⚠️ **Every transformation in the pipeline gets a one-line comment stating what timestamp its inputs are valid as-of.** This discipline is itself a talking point.

### 2.2 🧠 Alpha, Signal, and Factor

- **Signal:** a number, computed at each point in time, that we believe carries information about future returns. Ours is the FOMC sentiment Z-score.
- **Alpha:** the component of return that is *predictable* and *not explained by simple exposure to the market* (i.e., not just "the market went up"). A signal "has alpha" if it predicts returns beyond what you'd get by buying and holding.
- **Factor:** a systematic driver of returns shared across many assets (e.g., value, momentum, size). Our single-asset signal isn't a full factor, but the vocabulary overlaps.

### 2.3 🧠 FinBERT

- **BERT** ("Bidirectional Encoder Representations from Transformers") is a language model that reads text and produces context-aware numerical representations. "Bidirectional" = it looks at words to the left *and* right simultaneously.
- **Fine-tuning** = taking a general pre-trained model and continuing to train it on a narrow, labeled dataset so it specializes.
- **FinBERT** = BERT fine-tuned on **financial text** (the `ProsusAI/finbert` checkpoint was trained on financial news / the Financial PhraseBank). It outputs, for a piece of text, three probabilities: **P(positive), P(negative), P(neutral)**.

**Why it matters:** general BERT sees *"the company cut its dividend"* as ordinary words. FinBERT has learned that "cut its dividend" is a strongly negative financial event. Domain adaptation is the whole reason we don't use a generic sentiment model.

⚠️ **Caveat we will confront honestly:** FinBERT was trained on *analyst/news* language, not *central-bank* language. Fed prose is uniquely hedged and euphemistic ("modestly," "somewhat," "attentive to risks"). Part of our diagnostics (the "Fed-speak confusion matrix," Section 8) exists precisely to check whether FinBERT's notion of sentiment transfers to Fed-speak — a genuine research question, not a formality.

⚠️ **The label-ordering trap (verified, Phase 0).** The `ProsusAI/finbert` checkpoint declares:

```python
model.config.id2label == {0: "positive", 1: "negative", 2: "neutral"}
```

This is **neither alphabetical** (which would be negative/neutral/positive) **nor** the ordering most sentiment checkpoints use. Writing `logits[:, 0]` and calling it "negative" — the natural guess, and what most tutorial code does — **silently inverts the entire signal.** The pipeline would run perfectly, produce plausible-looking numbers, and give you an IC with the wrong sign. Nothing would crash.

**Rule:** always read the mapping from `model.config.id2label` at load time and index by *name*, never by position. `tests/test_sentiment_shapes.py::test_finbert_label_ordering_is_as_expected` pins this so a future checkpoint revision cannot change it unnoticed.

🧠 **A first, concrete data point on the out-of-domain question.** Scoring the archetypal hawkish sentence during Phase 0 environment validation:

> *"The Committee remains highly attentive to inflation risks."*
> → P(neutral) = **0.642**, P(positive) = **0.337**, P(negative) = **0.021**

FinBERT reads an *inflation-vigilance* sentence — unambiguously hawkish to a macro analyst, and therefore a headwind for equities — as mildly **positive**. One sentence is an anecdote, not evidence. But it is precisely the failure mode §8.3 exists to quantify, and it is a live illustration of the sign trap in §2.7: FinBERT's axis is *sentiment valence*, not *equity-directional implication*, and those two axes are not the same axis.

### 2.4 🧠 Information Coefficient (IC)

**Definition:** the correlation between the signal's prediction at time *t* and the actual asset return over the following horizon *h*.

$$\text{IC} = \text{corr}\big(\text{signal}_t,\; R_{t \to t+h}\big)$$

- **Pearson IC** uses raw values (sensitive to outliers and to the exact magnitude).
- **Spearman (rank) IC** uses *ranks* (robust; asks only "did higher signal go with higher return?"). We report **Spearman** as primary because it's robust to the fat tails and outliers endemic to return data, and Pearson as a secondary cross-check.

**Interpretation / scale (this surprises people):**
- IC ≈ **0.05 (5%)** is considered a **strong, institution-grade** signal in cross-sectional equity research.
- IC of 0.10+ in a clean, large-sample setting is exceptional (and should make you suspicious of a bug).
- Our setting is **time-series, single-asset, tiny-sample** (≈8 events/year), so ICs will be noisy and the honest deliverable is *"IC = x, with a t-statistic of y and a 95% confidence interval of [a, b]."* The confidence interval matters more than the point estimate.

💬 *"IC is my primary metric because it's a direct measure of predictive correlation, decoupled from the position-sizing and leverage choices that make Sharpe ratios so easy to inflate."*

### 2.5 🧠 IC Decay (alpha half-life)

**Definition:** how IC changes as you extend the prediction horizon *h* (1 day, 3, 5, 10, 20 days).

**Why it's profound:** it reveals *how fast the market prices in* the information. A signal whose IC is high at *h=1* but collapses by *h=5* is a fast, micro-structure-flavored edge (and will be eaten by trading costs). A signal that stays flat across horizons is either very persistent... or a sign your returns are overlapping and autocorrelated (a bug to check). Plotting decay proves you think about *the economics of information*, not just a single magic number.

### 2.6 🧠 The metrics of a backtest

- **Return / CAGR:** compounded annual growth rate of the strategy.
- **Volatility:** annualized standard deviation of returns — a proxy for risk.
- **Sharpe Ratio:** $\dfrac{\text{annualized excess return}}{\text{annualized volatility}}$. Reward per unit of risk. A Sharpe around 1 is good for a single strategy; 2+ is excellent; **anything above ~3 in a student backtest is almost always a bug or overfit.**
- **Maximum Drawdown (MDD):** the largest peak-to-trough loss. Answers "how much pain would I have endured?"
- **Win Rate:** fraction of trades that were profitable. (Deceptive on its own — a 40%-win-rate strategy can be very profitable if wins are bigger than losses.)
- **Turnover:** how much you trade. Drives transaction costs.
- **Basis point (bp):** 1 bp = 0.01% = 0.0001. Transaction costs and spreads are quoted in bps.

### 2.7 🧠 Hawkish vs. Dovish (the domain semantics)

- **Hawkish** = leaning toward *tighter* policy (higher rates / fighting inflation). Generally a **headwind** for equities. Think: a hawk, aggressive.
- **Dovish** = leaning toward *easier* policy (lower rates / supporting growth/employment). Generally a **tailwind** for equities. Think: a dove, gentle.

⚠️ **The sign trap:** FinBERT's "positive/negative" is about *sentiment*, not about *equities*. A hawkish statement might contain "positive" language about a *strong economy* — which can be *bearish* for stocks (because it implies rate hikes). We must not assume "positive sentiment → stocks up." We let the **data** determine the sign of the relationship (via IC), and we interpret the result through the hawkish/dovish lens. This nuance is gold in an interview.

---

## 3. The data reality of the FOMC

Before any code, understand the *actual* data-generating process. Getting this wrong is how look-ahead bias enters.

### 3.1 The FOMC calendar

- The FOMC holds **8 scheduled meetings per year** (occasionally an unscheduled/emergency meeting).
- Meetings are typically **two days**; the decision is announced on **day 2**.
- Archives on federalreserve.gov go back to the **1990s**.

### 3.1.1 🔧 The disclosure regime changed — measured in Phase 1

The original text above said a clean series starts "roughly 1994–2000 onward." That is too vague, and the vagueness hides the single most consequential data fact in the project. **Measured statement counts per year:**

```
1994:  6    2000:  8    2007: 10    2014:  8    2021:  8
1995:  3    2001: 11    2008: 11    2015:  8    2022:  8
1996:  0 ⚠  2002:  8    2009:  8    2016:  8    2023:  8
1997:  1    2003:  8    2010:  9    2017:  8    2024:  8
1998:  3    2004:  8    2011:  8    2018:  8    2025:  8
1999:  6    2005:  8    2012:  8    2019:  9    2026:  5 (YTD)
            2006:  8    2013:  8    2020: 10
```

**Definition — disclosure regime:** the central bank's own policy about *when* it tells the public anything. It is not a market variable; it is an institutional choice, and it changed underneath our data.

- **1994–1998:** the FOMC issued a statement **only when it changed policy.** No policy change meant complete silence. 1996 had no rate moves, hence zero documents.
- **From May 1999:** a statement after *every* meeting. The count stabilises at 8/year from 2000.

🧠 **Why this forces the sample to start in 2000 — two independent reasons.**

**(1) Selection effect.** Before 1999 the *existence* of a statement is itself the signal, and a far louder one than its tone: a statement existing means "policy just moved." Those years therefore have a different data-generating process, and pooling them risks a spurious positive IC driven by the existence effect that we would then misattribute to FinBERT reading tone.

| Era | What carries the information |
|---|---|
| 1994–1998 | **Existence** (statement ⟺ policy changed). Tone is secondary. |
| 1999/2000– | **Tone only.** A statement always exists; the surprise is in the words. |

**(2) The Z-score silently breaks.** §4.2's trailing window assumes roughly regular event spacing — "*L* = 6 meetings" is meant to proxy "the last ~9 months of policy." With 1996 empty, a 6-meeting lookback at the 1997-03 statement reaches back to **1994-11**: a 28-month window masquerading as a 9-month one, describing an entirely different policy era. Nothing errors; the signal's core premise just quietly stops holding.

**The cost of excluding them is negligible**, which is what makes the decision easy: 19 events (8% of the corpus), moving the minimum detectable IC from 0.127 to 0.132. A rounding error, traded for a homogeneous data-generating process.

💬 *"I found the Fed only announced policy CHANGES until 1999 — 1996 has zero statements. That's a selection effect, and it also breaks my trailing Z-score, because a 6-meeting lookback in 1997 reaches back to 1994. I cut the sample to 2000+ and it cost me 8% of the data."*

### 3.2 The three document types (and their VERY different timestamps)

| Document | Released | Release time | Look-ahead risk |
|---|---|---|---|
| **Policy statement** | Meeting day (day 2) | **2:00 PM ET** (2:15 PM before 2013) | Must trade at/after release → we use **next open** |
| **Meeting minutes** | **~3 weeks after** the meeting | 2:00 PM ET | **HIGH** — must stamp with *release* date, never meeting date |
| **Press-conference transcript** | Meeting day (post-2:30 PM) | afternoon | same-day, intraday — we avoid unless doing intraday |

**Design decision:** **Phase 1 will start with the policy statements only.** They are the cleanest (short, one per meeting, unambiguous same-day afternoon timestamp). The minutes are a valuable *second* signal we add once the statement pipeline is proven — but they carry the 3-week-lag trap, so we treat their release date as a first-class, separately-scraped field. *Never infer a minutes timestamp from the meeting it describes.*

🔧 **Two release-time details established in Phase 0**, now encoded in `config.yaml`:

**(a) The scheduled time changed.** Statements from 1994–2012 were released at roughly **2:15 PM ET**; from 2013 onward at **2:00 PM ET**. Under our "next trading open" entry rule this distinction *never changes an entry date* — both are after the 9:30 AM open and before the 4:00 PM close, so both map to the same next session. We record it anyway, because a robustness variant that enters at the **same-day close** *would* care, and because knowing a detail you did not need is cheaper than discovering one you did.

**(b) Unscheduled statements do not follow the rule at all.** Two matter enormously and are hardcoded as auditable `manual_override` entries:

| Date | Actual release | Why it breaks naive logic |
|---|---|---|
| **2008-10-08** | **07:00 ET** | Coordinated global emergency cut, released **before the open**. A date-only rule cannot tell this apart from a 2 PM release, yet the tradable response is a full session earlier. |
| **2020-03-15** | **17:00 ET, a Sunday** | COVID cut to 0–0.25%. There *is* no "next calendar day" that is a trading day by coincidence alone — you must go through a real calendar. |

Each override is stamped `release_time_source = "manual_override"` (versus `"scheduled_1400ET"`) so the handful of hand-set timestamps are visible in the data rather than invisibly baked in. **Any number a human typed should be flagged as a number a human typed.**

💬 *"I separated the statement and minutes pipelines specifically because they have different release lags. Conflating them is the classic way sentiment-on-Fed projects leak future information."*

### 3.3 Timestamp alignment rule (the heart of leak-prevention)

For each document we record two dates:
1. `event_date` — the calendar date the *content* refers to (meeting date). **For display/joining only.**
2. `release_datetime` — the exact date+time the text became **public**. **This is the only timestamp allowed to drive trading.**

**The alignment rule (enforced in code):**

> A signal computed from a document is *effective* at the **open of the first trading day strictly after `release_datetime`**. We trade at that open and hold for horizon *h*.

Why "strictly after" and "next open" rather than "same-day close"? Two reasons: (a) it is the most conservative, unambiguously-implementable rule (no intraday timing assumptions, no debate about whether you could really hit the 4 PM close); (b) it makes the return series cleanly non-overlapping and easy to reason about. We can *relax* to "same-day close" later as a robustness variant and compare — but we **start strict.**

🔧 **How the rule is implemented (decided in Phase 0): a single `searchsorted`.**

**Definition — `searchsorted`:** a binary search over a *sorted* array that returns the index at which a new key would be inserted to keep the array sorted. With `side="right"`, ties resolve to the right, so the returned index points at the first element **strictly greater than** the key.

That is our alignment rule, verbatim, in one line:

```python
idx = trading_days.searchsorted(release_datetime, side="right")
entry_date = trading_days[idx]
```

No loop, no `+ 1`, no branch for weekends, no off-by-one to argue about in an interview. The rule and the code are the same sentence.

⚠️ **Why `release_date + pd.Timedelta(days=1)` is not an acceptable shortcut.** It is wrong on exactly the cases that matter most — the emergency meetings, which are also the highest-volatility, highest-signal events in the sample:

- **2020-03-15** was a **Sunday**. Date arithmetic gives Monday 2020-03-16, which happens to be a trading day — *by luck*. The same code applied to a Friday-evening release gives Saturday, and then you need special-casing, and then you have written a worse trading calendar by accident.
- **2008-10-08 07:00 ET** was released **before that day's open**. Date arithmetic cannot even represent the question, because it has thrown away the time. A correctly-typed timestamp makes the pre-open case *visible*; a date makes it *invisible*.

🧠 **The general principle:** `release_datetime` is a **timezone-aware `America/New_York` timestamp**, never a date. Naive datetimes (no timezone attached) are a bug farm in market data — the moment anything touches UTC, a 2:00 PM ET release silently becomes 7:00 PM UTC, crosses midnight in some conversions, and shifts a day. Attach the timezone once, at the point of creation, and never strip it. `next_trading_open()` **raises** on naive input rather than guessing.

### 3.3.1 🔧 Refinement forced by Phase 1: compare against the session OPEN, not the date

The rule above says "the first trading **day** strictly after the release." That silently assumes every release lands during or after a session. **Several do not:**

| Date | Release | Consequence of a day-granularity rule |
|---|---|---|
| 2008-10-08 | **07:00 ET** | coordinated emergency cut, **pre-open** — entered a full session late |
| 2020-03-23 | **08:00 ET** | pre-open — entered a session late |
| 2020-03-15 | 17:00 ET, **Sunday** | no "next calendar day" that is a session except by luck |

Entering a session late is not a rounding error here: these are the highest-impact events in the sample, so the error concentrates exactly where it costs most.

**The fix needs no special-casing at all** — just compare against the right instant. The NYSE opens at **09:30 ET**, so:

- a **14:00** release has *missed* that day's 09:30 open → entry is the **next** open ✅ (unchanged for every ordinary statement)
- an **08:00** release has *not* → entry is **that day's** open ✅

Same rule, correctly applied. Implemented by localising each session date to its 09:30 opening instant and `searchsorted`-ing the release timestamp into that array.

🔧 **And the release times are now *parsed from the documents themselves*, not assumed.** FOMC statements state their own release time in the text:

> `For release at 2:00 p.m. EDT` · `For release at 5:00 p.m. EDT` (2020-03-15)

`extract_release_time()` reads it. **88 of the 225 sampled statements now carry an evidence-based timestamp** rather than a convention-based one. Both hand-entered overrides were independently **confirmed** by the documents' own text:

```
2008-10-08  override 07:00  document says 07:00  -> AGREE
2020-03-15  override 17:00  document says 17:00  -> AGREE
```

⚠️ **The safety principle, and why precedence is ordered the way it is.** Where the document is silent (older statements say only "For immediate release"), the **14:00 default** applies — and the config schema *enforces* that this default is after the session open. That is deliberate:

> **Erring late costs signal. Erring early fabricates alpha.**

An unverified document must never win same-session entry. Precedence is `manual_override` → `parsed_from_document` → `scheduled_default`, so a regex change can never silently overturn a researched value, and any disagreement between the two is reported rather than buried.

**Measured result on the real panel:** smallest release→entry lag **1.50 hours** (an 08:00 release to the 09:30 open — exactly right), median **19.5 hours**, all 225 rows strictly positive.

### 3.4 Price data (SPY)

- **SPY** is the SPDR S&P 500 ETF Trust — the most liquid proxy for the US large-cap equity market. It began trading **1993-01-22**, which conveniently pre-dates the first regular FOMC statement (February 1994), so the price series never constrains the sample.
- We pull **daily OHLC** (Open/High/Low/Close) via `yfinance`.

**Definition — ETF (Exchange-Traded Fund):** a fund that holds a basket of securities and whose *shares themselves* trade on an exchange like a single stock. SPY holds the S&P 500 constituents, so buying one SPY share is economically equivalent to buying a slice of all 500 companies — with a single, continuously-quoted price, which is what makes it usable as "the market" in a backtest.

⚠️ 🔧 **Adjusted vs. unadjusted prices — this section originally contained an error, corrected in Phase 0.**

**Definition — price adjustment:** rewriting historical prices so that dividends and splits do not appear as price gaps. When SPY pays a $1.50 dividend, its price drops ~$1.50 overnight. That is not a loss — you received the cash — but an unadjusted price series records it as one. SPY distributes roughly **1.3% per year** across four quarterly dividends; over the 30-year sample that is a very large cumulative distortion.

**The original error:** Appendix A specified `price_field: "Adj Close"` while §4.3 (below) specifies entering at the **Adjusted Open**. These are incompatible. With `yfinance`'s `auto_adjust=False`, you receive **raw** `Open/High/Low/Close` *plus one extra* `Adj Close` column. **There is no `Adj Open`.** Entering on a raw `Open` and exiting on an `Adj Close` mixes two different price bases, and injects every dividend in the holding window into the return as a phantom loss.

**The fix:** use **`auto_adjust=True`** (now the `yfinance` default), which back-adjusts **all four** OHLC columns on a single consistent basis. Then `Open` *is* the adjusted open, and an open-to-open return is clean and executable. `config.yaml` therefore carries `auto_adjust: true`, `entry_price_field: "Open"`, `exit_price_field: "Open"`, and `test_config.py::test_price_fields_are_consistent` fails if anyone sets `auto_adjust: false` while still trading on `Open`.

💬 *"I caught an internal inconsistency in my own spec — I'd written 'adjusted open' in the methodology and 'Adj Close' in the config, and yfinance doesn't actually give you an adjusted open unless you turn on full auto-adjustment. Mixing the two bases would have leaked every dividend into my returns as a fake loss."*

⚠️ **Weekends/holidays:** FOMC releases can land on any weekday, and markets close on holidays. "Next trading day" must be computed against an actual **trading calendar**. We use the **SPY price index itself as the source of truth for "is this a trading day."**

🧠 **Why the price index and not an external calendar?** It is *self-consistent*: a day we cannot price is, by definition, a day we cannot trade. An external calendar could assert that a session existed for which our data has no bar, and we would then be unable to compute the entry price for a trade the calendar says we made.

We still install `pandas_market_calendars` — but as a **validator, not an authority**. `prices.validate_against_nyse()` compares our index against the official NYSE session list in both directions:
- NYSE sessions **missing** from our data → a download gap. Silently shifts an entry date by one and corrupts every downstream return.
- Price rows on **non-NYSE dates** → a data error from the provider.

An empty discrepancy frame is the pass condition. This is a general pattern worth internalising: **pick one source of truth, then use the second source to audit it — never to silently fill it in.**

### 3.5 Sample-size reality check (say this out loud in interviews)

🔧 **MEASURED in Phase 1 — the estimate has been replaced by the count.**

The original text said "≈ 240". A later revision estimated "≈ 263". The scraper has now been run, and the real funnel is:

| Step | Count | Why |
|---|---|---|
| documents discovered across all index pages | 393 | statements + minutes, 1994 → 2026 |
| of which **statements** | **244** | the rest are 149 minutes (Phase 5) |
| after dropping flagged parses | 241 | −3, all 1994–1995, all outside the sample anyway |
| **after the 2000-01-01 disclosure-regime cut** | **225** | −16, the sparse-disclosure era (§3.1) |
| after the unscheduled filter (`include_unscheduled: true`) | 225 | −0, unscheduled events are kept and tagged |
| **usable observations for the IC, `n`** | **219** | minus the first *L* = 6 (undefined Z-score window, §4.2) |

Composition: **212 scheduled, 13 unscheduled**, spanning **2000-02-02 → 2026-07-29**.

🔧 **The disclosure-regime discovery (§3.1) is the reason 244 statements yield only 225.** From 1994 to 1998 the FOMC issued a statement *only when it changed policy* — 1996 has **zero**, 1997 has **one**. Those years are a different data-generating process and are excluded from the primary sample; they remain scraped and available as a labelled robustness check.

⚠️ **Report `n`, never assume it.** A statement that fails to parse must be *dropped and counted*, never silently scored as neutral. `scripts/build_panel.py` prints the funnel above on every run, and asserts `n_scraped == n_parsed + n_flagged`.

That is still a **tiny dataset** by ML standards. Consequences we must respect:
- We cannot "train" a model on this; we can only **evaluate a pre-trained model's signal.** (Good — that's the plan.)
- Every statistic has wide error bars. **Confidence intervals and t-stats are mandatory, not optional.**
- Overfitting risk is enormous: with ~263 points, if you try 20 different signal variants and pick the best, you *will* find a "significant" result by chance. Section 5 addresses this head-on.

🧠 **Put the number in perspective.** At the measured *n* = 219, the standard error of a correlation near zero is approximately $1/\sqrt{n} \approx 0.068$. So the **smallest IC that could ever reach statistical significance at the 5% level is about $1.96 \times 0.068 \approx 0.13$** — and §2.4 tells us that an IC of 0.05 is already *institution-grade*. In other words:

> **This sample is too small to detect a genuinely good signal.** A real, tradable, professional-quality edge of IC ≈ 0.05 would be statistically indistinguishable from zero here.

This is not a flaw in the project — it is *the single most important finding the project can produce*, and it must be stated up front rather than discovered at the end. It reframes the whole exercise: we are not asking "is there an edge?", we are asking **"what can and cannot be learned from 219 observations?"** — which is a far more sophisticated question, and the one a research desk actually lives with. See §14.5 for the full power calculation.

💬 *"The binding constraint on this project isn't compute or data engineering — it's statistical power. With 219 events the minimum detectable IC is about 0.13, but a genuinely strong signal is 0.05. So I designed the evaluation around confidence intervals and a power analysis rather than a single hero backtest, and I can tell you exactly what sample size would be needed to answer the question properly."*

---

## 4. The mathematical methodology

We go from text → number in three steps, then evaluate.

### 4.1 Step A — Document sentiment aggregation ($S_t$)

**Problem:** FinBERT has a hard **512-token limit** (~a paragraph). A FOMC statement is several paragraphs; minutes are many pages. We must split, score, and aggregate.

**Procedure:**
1. Split the document into sentences with `nltk.tokenize.sent_tokenize` (or a robust regex fallback).
2. Score each sentence *i* with FinBERT → probabilities $(p^+_i, p^-_i, p^0_i)$ and an arg-max label.
3. Aggregate to one score per document.

**Two aggregation formulas — we implement both and compare:**

*(a) The count-based score (as in the original spec — simple, interpretable):*

$$S_t = \frac{N_{\text{positive}} - N_{\text{negative}}}{N_{\text{total sentences}}}$$

where $N_{\text{positive}}$ is the number of sentences whose arg-max label is positive, etc. Ranges in $[-1, 1]$.

*(b) The probability-weighted (continuous) score — more information-dense, less lossy:*

$$S_t^{\text{prob}} = \frac{1}{N}\sum_{i=1}^{N} \big(p^+_i - p^-_i\big)$$

🧠 **Why offer both?** The count version throws away confidence (a 51%-positive sentence counts the same as a 99%-positive one). The probability version keeps it. But the count version is more robust to FinBERT being *slightly* miscalibrated on Fed-speak. We compute both, and report which gives a cleaner IC. **Deciding between them is itself a research finding**, not a coin flip — and we must fix the choice *before* looking at returns to avoid cherry-picking (Section 5).

⚠️ **Boilerplate contamination:** FOMC statements contain repeated procedural sentences ("Voting for the monetary policy action were..."). These are sentiment-neutral noise. A refinement (Phase 2 stretch goal): strip known boilerplate / vote-tally lines before scoring, and document the effect.

### 4.2 Step B — The Sentiment Delta / rolling Z-score ($Z_t$)

🧠 **The key economic insight:** markets react to **surprises**, not **levels**. The Fed is *structurally* cautious; its baseline tone is always somewhat hawkish/hedged. What moves markets is a statement being **more or less** hawkish *than recently*. So we normalize each raw score against its own recent history:

$$Z_t = \frac{S_t - \mu_{t}^{\text{(lookback)}}}{\sigma_{t}^{\text{(lookback)}}}$$

where $\mu_t$ and $\sigma_t$ are the mean and standard deviation of $S$ over the **previous** *L* meetings (e.g., *L* = 6 or 12).

⚠️ **The look-ahead trap in Z-scoring (critical):** the mean and std MUST be computed on a **trailing window that excludes the current observation**, i.e., using only meetings *strictly before* *t*. In pandas terms:
```python
mu = S.rolling(L).mean().shift(1)  # .shift(1) => exclude current point
sigma = S.rolling(L).std().shift(1)
Z = (S - mu) / sigma
```
Forgetting the `.shift(1)` (or using pandas' default centered/expanding stats that peek at the current value) is a textbook leak. We will **unit-test** this: the Z at time *t* must be reproducible using only rows `< t`.

🧠 **Why Z-score and not raw $S_t$?** Three reasons: (1) removes the persistent baseline tone; (2) makes the signal **stationary-ish** and comparable across regimes (the Fed of 2008 vs. 2021); (3) puts the signal on a natural, unit-free scale so a threshold like "±1 standard deviation" is meaningful. The cost: the first *L* meetings have no signal (undefined window) — we discard them, and we lose a little sample. Worth it.

**Design knobs (to be fixed before evaluation, see Section 5):** window length *L* ∈ {6, 12}, and whether *S* is the count or probability version. We pre-register a **primary** configuration and treat the rest as robustness checks.

### 4.3 Step C — Forward returns ($R_{t\to t+h}$)

The realized return of SPY from the entry point over horizon *h* trading days:

$$R_{t\to t+h} = \frac{P_{t+h} - P_{t}}{P_{t}}$$

where — per the alignment rule (3.3) — $P_t$ is the **adjusted Open of the first trading day after the release**, and $P_{t+h}$ is the **adjusted Open** *h* trading days later.

🔧 **Both legs use the same price field.** The original text said "the Adjusted price *h* trading days later," which is ambiguous; §3.4 explains why mixing an `Open` entry with a `Close` exit (or a raw price with an adjusted one) corrupts the return. Open-to-open is:
- **executable** — you can actually transact at an opening auction;
- **consistent** — the same field, the same adjustment basis, on both legs;
- **conservative** — it excludes the overnight gap immediately following the release, which is the part of the move you could *not* have captured under the strict entry rule.

**Definition — trading-day horizon:** *h* counts **sessions, not calendar days**. *h* = 5 means five *trading* days ahead — roughly a calendar week, but it steps over weekends and holidays automatically because we index into the trading-day array rather than adding timedeltas. This is the same discipline as the entry rule, applied to the exit.

We compute a small **panel** of forward returns for *h* ∈ {1, 3, 5, 10, 20} to drive the IC-decay analysis.

**Definition — panel:** a table with one row per *observation* (here, per FOMC event) and one column per *variable* (here, each horizon's forward return, plus the signal and metadata). Holding all horizons side by side in one table — rather than recomputing per horizon — is what makes the IC grid in §5.1 a single vectorised operation instead of five separate pipelines that could drift apart.

⚠️ 🔧 **Overlap warning — the original text here was WRONG, and Phase 1 measured it.**

The original claim was: *"With ~8 events/year (~32 trading days apart) and h ≤ 20, overlap is minimal — note it and move on."* That reasoning uses the **average** spacing and ignores the **minimum**. Measured on the real 225-event sample:

| | sessions |
|---|---|
| median gap between events | 30 |
| **minimum gap** | **3** |

**Overlapping pairs, by horizon:**

| *h* | overlapping pairs | verdict |
|---|---|---|
| 1 | 0 | clean |
| 3 | 0 | clean |
| **5 (primary)** | **1** | 2007-08-08 → 2007-08-13 |
| 10 | 8 | material |
| 20 | **16** | **substantial — 7% of the sample** |

**Why the average was misleading:** unscheduled meetings cluster *inside* the gaps between scheduled ones, and they cluster precisely during crises — 2007-08, 2008-01, 2008-03, 2008-10, 2020-03. So the overlap is not spread evenly; it is concentrated in the highest-volatility, highest-signal episodes, which is the worst place for it.

**Why it matters (PLAN.md §14.4):** the IC t-statistic assumes independent observations. Overlapping windows share price moves, so returns are autocorrelated, the *effective* sample is smaller than *n*, and the t-statistic is **inflated** — significance you have not earned.

**What we do about it, in order of preference:**
1. **Report it.** `check_overlap()` enumerates every offending pair at every horizon; the number goes in the results table, not a footnote.
2. **Prefer short horizons for the headline.** *h* ∈ {1, 3} are provably clean; *h* = 5 has a single pair.
3. **Newey–West standard errors** at *h* ≥ 10 — an estimator that stays valid under autocorrelation (hence `statsmodels` in the dependency list).
4. **Robustness check:** rerun with `include_unscheduled: false`, which removes most of the clustering, and compare.

💬 *"My plan asserted overlap was negligible because events average 32 sessions apart. When I measured it, the minimum gap was 3 sessions and 16 pairs overlapped at h=20 — because emergency meetings cluster inside the gaps, during exactly the crises that dominate the return variance. So I moved my headline to short horizons and used Newey–West at the long ones."*

### 4.4 Step D — The trading rule (signal → position)

A deliberately **simple, transparent** rule (complexity here just invites overfitting):

$$\text{position}_t = \begin{cases} +1 & \text{if } Z_t > +\theta \quad (\text{go long SPY}) \\ -1 & \text{if } Z_t < -\theta \quad (\text{go short SPY}) \\ \;\;0 & \text{otherwise (hold cash)} \end{cases}$$

with threshold $\theta = 1.0$ as the baseline.

⚠️ **The sign is an empirical question, not an assumption.** Before committing to "positive Z → long," we check the *sign of the IC*. If the data say more-dovish-than-expected → SPY up, then positive-sentiment-shift → long is correct. If FinBERT's "positive" correlates with hawkish-and-therefore-bearish, we may need to **flip** the rule — but we fix the sign convention from a **hypothesis** (dovish surprise is bullish) and then *test* it, rather than fitting the sign to maximize returns (which would be overfitting). Document the reasoning explicitly.

Strategy return in each period: $r^{\text{strat}}_{t} = \text{position}_t \times R_{t\to t+h}$ (minus costs, Section 8.2).

---

## 5. Statistical rigor: why most backtests are lies

**This is the section that separates you from every other candidate.** Read it twice.

### 5.1 🧠 The overfitting / multiple-testing problem

With ~263 usable events (§3.5) and many knobs (aggregation method, window *L*, threshold *θ*, horizon *h*, statement-vs-minutes, sign), the number of distinct strategies you *could* test easily exceeds 100. **If you try 100 random strategies on noise, ~5 will look "significant at 5%" by pure chance.** Picking the best-looking one and presenting it is called **data snooping / p-hacking**, and it's why so many published anomalies don't replicate.

**Defenses we adopt:**

1. **Pre-registration (informal).** Before running the backtest, we *write down* in the notebook: primary aggregation = probability-weighted; primary *L* = 6; primary *θ* = 1.0; primary horizon for the trade = 5 days; primary hypothesis = "dovish surprise (positive sentiment shift) is bullish." Everything else is explicitly labeled a **robustness check**, not a competing candidate for "the result."

2. **Report the whole grid, not the max.** We show IC across *all* horizons and both aggregation methods as a **heatmap/table** — so the reader sees the full distribution of outcomes, not a cherry. A signal that's positive across many horizons is credible; one that spikes at a single lucky *h* is not.

3. **Deflated / Haircut Sharpe intuition.** We won't over-engineer this, but we'll *state* that the naive Sharpe should be discounted for the number of trials, and we keep the trial count small and disclosed.

💬 *"I pre-registered a primary configuration and reported the full parameter grid rather than the best cell, so the reader can see I'm not cherry-picking a lucky horizon."*

### 5.2 🧠 Is the IC statistically distinguishable from zero?

A point estimate like "IC = 0.06" is meaningless without an error bar. Two tools:

- **t-statistic of the IC.** For *n* observations, $t \approx \text{IC}\sqrt{\dfrac{n-2}{1-\text{IC}^2}}$. With *n* ≈ 200, an IC of 0.06 gives *t* ≈ 0.85 — **not significant.** This math is sobering and honest: *small samples make even "good-looking" ICs statistically weak.* Show it.

- **Bootstrap confidence interval.** Resample (signal, forward-return) pairs with replacement thousands of times, recompute IC each time, and report the 2.5th–97.5th percentiles. This makes zero assumptions about normality and is the most defensible error bar for a small, fat-tailed sample. **This is the single most impressive statistic you can show.**

### 5.3 🧠 Out-of-sample discipline (walk-forward)

Even though we don't *train* FinBERT, our *choices* (L, θ, sign, aggregation) are fit to the data if we choose them by looking at the whole history. Best practice:

- **Walk-forward / expanding window:** choose parameters using data up to year *Y*, evaluate on year *Y+1*, roll forward. This simulates how you'd actually deploy.
- Given the tiny sample, a full walk-forward may leave too little data to be meaningful; at minimum we do a **single train/test split** (e.g., choose primary config conceptually a priori, then report results on a held-out later period like 2015–present) and *discuss* the limitation. Honesty about power beats a fake robust result.

### 5.4 🧠 Purging & embargo (concept to name-drop correctly)

In event-driven backtests, if a training window and a test window share overlapping return horizons, information leaks across the split. **Purging** removes training samples whose label window overlaps the test set; an **embargo** adds a small gap after the test set. With ~32-trading-day spacing and *h* ≤ 20 we have natural separation, but **naming these concepts and explaining why they barely bite here** shows genuine expertise (ref: Marcos López de Prado, *Advances in Financial Machine Learning*).

### 5.5 The honesty clause

If, after all this, the signal's IC confidence interval comfortably straddles zero, **the correct deliverable is: "no statistically significant edge detected at these horizons, given n≈263; here is the confidence interval, and here is the sample size we'd need to detect an edge of plausible magnitude."** That is a *strong* result. Do not massage the pipeline until a positive appears — that process *is* the overfitting we warned about.

🔧 **And we can now be sharper than "here's what sample size we'd need," because §14.5 computes it in advance.** At *n* ≈ 263 the minimum detectable IC is ≈ 0.12, while an institution-grade signal is ≈ 0.05. **The study is underpowered by roughly an order of magnitude, and we know this before running it.** That converts the honesty clause from a graceful concession into a *prediction*: a null result is the expected outcome under almost any realistic effect size, so it tells you about the design rather than about the Fed. The genuinely interesting question then becomes the one in §14.5 — how to buy breadth instead of history.

---

## 6. Repository & environment architecture

### 6.1 Directory layout

🔧 **This is the layout as actually built** (2026-08-04), which differs from the original sketch in four places, each marked and explained below. The repository root is `C:\Users\Aryan\Work\QRT`.

```
QRT/                                 # repo root (git init'd on `main`)
├── PLAN.md                          # this document — the complete picture
├── STATUS.md                        # 🔧 NEW: the live progress ledger
├── README.md                        # how to run it; setup, CUDA notes, conventions
├── CLAUDE.md                        # 🔧 NEW: working agreements for AI-assisted sessions
│
├── pyproject.toml                   # 🔧 deps + cu128 torch index + pytest/ruff config
├── uv.lock                          # 🔧 NEW: exact resolved dependency graph
├── requirements.txt                 # exported from uv.lock for conventional readers
├── config.yaml                      # every knob; the pre-registration record (§5.1)
├── conftest.py                      # 🔧 NEW: makes `import src...` work under pytest
├── .gitignore                       # data/, .venv/, __pycache__, *.ckpt, local settings
├── .gitattributes                   # 🔧 NEW: eol=lf so the repo is identical on Linux
│
├── data/                            # gitignored; fully regenerable from src/
│   ├── raw/                         # IMMUTABLE — written once, never edited
│   │   ├── manifest.csv             # 🔧 NEW: URL, sha256, bytes, fetch time per file
│   │   ├── fomc_statements/         # scraped .htm, filename keyed by event date
│   │   └── fomc_minutes/            # Phase 5; filename = RELEASE date, not meeting date
│   ├── interim/
│   │   └── documents.parquet        # cleaned text + event_date + release_datetime
│   └── processed/
│       ├── spy_prices.parquet       # yfinance OHLCV, auto-adjusted
│       ├── sentiment_scores.csv     # S_t per document — the expensive cache
│       ├── sentence_scores.parquet  # one row per sentence — feeds the §8.3 audit
│       └── panel.parquet            # the join: signal, Z, position, forward returns
│
├── src/
│   ├── __init__.py                  # module map + the alpha_signal naming rationale
│   ├── config.py                    # ✅ pydantic-validated loader for config.yaml
│   ├── scrape_fomc.py               # Phase 1: index-crawling scraper + provenance
│   ├── prices.py                    # Phase 1: yfinance loader + NYSE cross-check
│   ├── align.py                     # Phase 1: release → next-trading-open (leak guard)
│   ├── sentiment.py                 # Phase 2: FinBERT engine
│   ├── alpha_signal.py              # 🔧 RENAMED from signal.py — Phase 3: rolling Z
│   ├── backtest.py                  # Phase 4: vectorised backtest + IC + bootstrap
│   └── diagnostics.py               # Phase 4: IC decay, cost curve, Fed-speak audit
│
├── scripts/
│   └── check_gpu.py                 # 🔧 NEW: proves a real CUDA kernel launches
│
├── notebooks/
│   └── 99_research_report.ipynb     # Phase 4: the final narrative deliverable
│
├── reports/                         # 🔧 NEW: generated plots + tearsheet HTML
│
└── tests/
    ├── test_environment.py          # 🔧 NEW: Phase 0 DoD — imports + real GPU matmul
    ├── test_config.py               # 🔧 NEW: config validates; primary values pinned
    ├── test_align_no_lookahead.py   # leak canary — asserts signal_t uses only data < t
    ├── test_zscore_no_lookahead.py  # leak canary — the Z-score .shift(1) guard
    └── test_sentiment_shapes.py     # probability/score invariants + label ordering
```

#### 🔧 The four deviations, and why

**1. `src/signal.py` → `src/alpha_signal.py`.** `signal` is a **Python standard-library module** (process signal handling), and `torch` imports it during startup. A top-level `signal.py` on `sys.path` *shadows* the stdlib one, so `import torch` fails with an error raised from deep inside torch's initialisation that names neither your file nor the real problem. Renaming costs nothing and removes a class of bug that reliably consumes an afternoon.

🧠 **The general lesson — module shadowing.** Python resolves imports by walking `sys.path` in order, and the script's own directory usually comes first. Any file you name after a stdlib module (`signal.py`, `types.py`, `queue.py`, `random.py`, `email.py`, `json.py`…) will be found *before* the real one by every library in your process. The failure appears in someone else's code, which is why it is so hard to diagnose.

**2. A provenance manifest in `data/raw/`.** "Raw is immutable" is only a *claim* unless something can check it. `manifest.csv` records, per fetched file: source URL, HTTP status, byte count, **sha256 hash**, and UTC fetch timestamp.

**Definition — sha256:** a cryptographic hash — a fixed-length fingerprint of a byte sequence. Change one byte and the hash changes completely. Storing hashes lets `verify_manifest()` re-read every file and *prove* the corpus has not drifted, and lets a third party confirm their scrape matches ours byte-for-byte **without us shipping the bytes**. It is also the reason `fetch()` returns `bytes` rather than `str`: decoding to text requires guessing an encoding, and a guess makes the hash unstable.

**3. `conftest.py` at the root.** Its presence causes pytest to prepend the repository root to `sys.path`, which is what makes `from src.config import load_config` work without installing the project as a package. `pyproject.toml` sets `[tool.uv] package = false` — this is a research repository, not a distributable library, so there is nothing to build or publish.

**4. `scripts/check_gpu.py` and `tests/test_environment.py`.** Explained in §6.2 — the short version is that `torch.cuda.is_available()` returning `True` **does not mean CUDA works** on this GPU, and only an actual kernel launch settles it.

🧠 **Why separate `raw` → `interim` → `processed`?** This is the standard data-engineering pattern (cf. the *Cookiecutter Data Science* layout), and each stage has a distinct guarantee:

| Stage | Guarantee | Cost to rebuild |
|---|---|---|
| `raw/` | **Immutable.** Byte-identical to what the server sent. Never edited by hand. | Minutes of network, and it hits someone else's server — so: **once, ever.** |
| `interim/` | Cleaned and parsed, but not yet *interpreted*. Recoverable from `raw/` alone. | Seconds. Cheap to throw away and redo. |
| `processed/` | Model outputs and joins — the expensive, reusable results. | Minutes of GPU. Redone rarely. |

**Why the caching layer matters more than it looks.** `processed/sentiment_scores.csv` is the **expensive cache**: FinBERT over ~270 documents × dozens of sentences each. You will re-run the *backtest* dozens of times while exploring, and if each run re-scores the corpus, the iteration loop is minutes instead of milliseconds — and a slow loop does not merely waste time, it **changes what research you do**, because you stop trying ideas that feel expensive. Caching the model output is therefore a *research-quality* decision, not only an engineering one.

### 6.2 Environment & dependencies

🔧 **As actually built and verified on 2026-08-04.** Three things changed from the original: the Python version, the package manager, and the torch build. Each is explained below.

| | Original plan | **As built** | Why |
|---|---|---|---|
| Python | 3.11 | **3.12.12** | Better wheel coverage across the whole stack; already cached locally by `uv`. Not 3.13/3.14 — `quantstats` and parts of `statsmodels` still lag there. |
| Package manager | `pip` + `requirements.txt` | **`uv` + `pyproject.toml` + `uv.lock`** | Reproducibility (below), and it is the only way to declare the CUDA torch index *declaratively*. |
| torch | generic CPU wheel | **`torch 2.11.0+cu128`** | The GPU is Blackwell; earlier CUDA builds do not contain kernels for it (below). |

#### 6.2.1 🧠 Why a lockfile, and what one actually is

**Definition — lockfile:** a machine-generated file that pins the **exact version and content hash** of *every* package **and every transitive dependency** — that is, the dependencies of your dependencies, and theirs, recursively.

This matters because of a gap most people never notice. A hand-written line like `pandas>=2.0` does **not** pin anything: it is a *constraint*, and a resolver run six months from now legitimately satisfies it with a different version. Worse, it says nothing at all about the ~150 packages you never named but which got installed anyway.

§5 of this document rests on the claim that these results are **reproducible**. That claim is false under a bare `requirements.txt` and true under a lockfile. `uv.lock` is therefore not tooling preference — it is what makes the central methodological claim honest.

A conventional `requirements.txt` (536 lines, fully pinned) is exported for readers who expect one:

```bash
uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt
```

#### 6.2.2 ⚠️ CUDA on Blackwell — the trap that passes every preliminary check

The machine has an **RTX 5070 Ti**, 16 GB, driver 610.88 (CUDA UMD 13.3).

**Definition — compute capability (CC):** NVIDIA's versioning of a GPU's *instruction set*, written `sm_XX`. The RTX 5070 Ti is the GB203 die, **compute capability `sm_120`**.

**Definition — cubin:** pre-compiled GPU machine code targeting one specific `sm_XX`. A wheel ships a set of cubins.
**Definition — PTX:** a forward-compatible intermediate assembly the driver can JIT-compile for a *newer* architecture at runtime. Slow to start, and not always shipped.

**The trap:** PyTorch wheels built against **CUDA 12.6 and earlier contain no `sm_120` cubins.** Such a wheel will:

1. `pip install` cleanly — ✅
2. `import torch` without error — ✅
3. report `torch.cuda.is_available() == True` — ✅
4. report the correct device name — ✅
5. …and then **fail at the first actual computation** with `RuntimeError: CUDA error: no kernel image is available for execution on the device`.

Every check a reasonable person would run passes. That is what makes it costly.

**The fix, encoded declaratively in `pyproject.toml`:**

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true          # only packages that name this index may resolve from it

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

`explicit = true` is important: without it, `uv` would search the PyTorch index for **every** package and could silently pull an unexpected fork of, say, `numpy`.

**Verified result** (`uv run python scripts/check_gpu.py`):

```
torch             : 2.11.0+cu128
built with CUDA   : 12.8
device            : NVIDIA GeForce RTX 5070 Ti
compute capability: sm_120
wheel arch list   : sm_75, sm_80, sm_86, sm_90, sm_100, sm_120
  native cubins present for sm_120
PASS: kernel executed and returned a numerically correct result.
```

The last line is the one that matters. `scripts/check_gpu.py` and `tests/test_environment.py::test_cuda_matmul_actually_runs` **both launch a real matmul on the device and check the numerical result**, because that is the only check that distinguishes a working install from the failure mode above.

🧠 **Two more CUDA facts worth owning:**
- **No system CUDA Toolkit is required.** The CUDA *runtime* is bundled inside the wheel (~3 GB); only the *driver* comes from NVIDIA's installer.
- **Driver 13.3 running a 12.8 runtime is fine.** Newer-driver/older-runtime is the supported direction; the reverse is not.

#### 6.2.3 🔧 Is the GPU actually needed? An honest cost/benefit

| Workload | CPU | GPU |
|---|---|---|
| Phase 2: ~270 statements, ~15–20k sentences | 2–4 min | < 10 s |
| Phase 5: ~270 minutes documents, ~60–90k sentences | 15–30 min **per re-run** | ~30 s |
| Any model larger than FinBERT (DeBERTa-v3, embeddings, an LLM labeller for §8.3) | impractical | routine |

**For Phase 2 as specified, the GPU buys nothing.** It becomes genuinely load-bearing at Phase 5 and for any model upgrade — and, as noted in §6.1, keeping the iteration loop fast changes which experiments you are willing to try.

The code is therefore **device-agnostic**: `config.yaml` sets `device: auto`, which resolves to `cuda` when available and `cpu` otherwise. The repository runs unchanged on a reviewer's laptop, and `test_environment.py` *skips* rather than *fails* its GPU assertions when no device is present. Portability here is itself a talking point.

#### 6.2.4 The dependency stack as installed

```
# Data & numerics
pandas 2.x · numpy · pyarrow (parquet) · pyyaml
pydantic              # 🔧 NEW — typed config validation (§6.3)

# Market data
yfinance 1.5.2 · pandas_market_calendars    # the latter VALIDATES, never authors

# Scraping
requests · beautifulsoup4 · lxml

# NLP
torch 2.11.0+cu128    # from the cu128 index, not PyPI
transformers 5.14.1   # ⚠️ major version AHEAD of the planned >=4.40 — see below
nltk                  # sentence tokenizer; regex fallback if punkt is unavailable

# Stats & analytics
scipy (spearmanr, bootstrap) · scikit-learn · statsmodels · matplotlib · seaborn

# Notebook & dev
jupyterlab · ipywidgets · tqdm · pytest · pytest-cov · ruff
```

⚠️ **`transformers` resolved to 5.14.1**, a **major version ahead** of the planned `>=4.40`. FinBERT loads and runs correctly under it (verified in Phase 0), but a great deal of tutorial code written for the 4.x API will not transfer verbatim. Treat any 4.x snippet as a hypothesis to test, not a recipe to paste.

⚠️ **`quantstats` is deliberately NOT installed by default.** It is the most likely dependency to break (it historically pins narrow `pandas` ranges — this is in the risk register at §10), so it lives behind an optional extra:

```bash
uv sync --extra tearsheet
```

`backtest.performance_metrics()` hand-computes Sharpe, max drawdown, win rate and CAGR, so **the research result never depends on it.** `quantstats` is for the pretty tearsheet only, and the project survives its absence.

⚠️ **`nltk`'s `punkt` sentence-tokenizer data is a runtime download** and can fail behind a proxy. `sentiment.split_sentences()` therefore ships a **regex fallback**, so the pipeline never hard-depends on a network fetch at analysis time.

### 6.3 Configuration-driven design

Every magic number (window *L*, threshold *θ*, horizons, cost bps, date range, aggregation method) lives in `config.yaml`, **not** hard-coded. Why: (1) reproducibility, (2) trivial robustness sweeps, (3) it *documents* your pre-registered primary configuration in one auditable place.

#### 6.3.1 🔧 The config is *validated*, not merely loaded

`config.yaml` is parsed into a **pydantic** model (`src/config.py`), not a plain dictionary.

**Definition — pydantic:** a Python library that builds runtime-validated classes from type hints. A `BaseModel` subclass parses *and checks* its inputs at construction time and raises immediately, naming the offending field.

**Why this is not over-engineering.** §5.1 makes `config.yaml` the **auditable record of pre-registration** — the artifact that proves the parameters were chosen before the returns were seen. An artifact that can be *silently* wrong cannot serve that role. Consider misspelling the window length:

```yaml
signal:
  zscore_window_l: 12      # lowercase L — a completely plausible typo
```

With a plain dict and a defensive `.get("zscore_window_L", 6)`, this runs, silently uses 6, and you report a result attributed to a configuration you did not use. With `extra="forbid"`, it raises at startup and names the key.

The schema enforces, among others:

| Rule | Catches |
|---|---|
| `extra="forbid"` on every section | typo'd or obsolete keys |
| `frozen=True` | mutation of config mid-run, which would decouple results from the file on disk |
| `max_length <= 512` | exceeding BERT's positional-embedding table |
| `zscore_window_L >= 2` | a standard deviation over a single observation |
| `horizons_days` ascending | an IC-decay plot drawn in the wrong order |
| `primary_horizon_days ∈ horizons_days` | a headline result invisible in the reported grid |
| `auto_adjust` must be true if trading on `Open` | the adjusted-price bug from §3.4 |

**And the pre-registration itself is a test.** `tests/test_config.py::test_preregistered_primary_values` pins every primary value. Changing one *after* seeing results makes the suite fail — which converts a quiet edit into a deliberate, reviewed act. That is exactly the friction pre-registration is supposed to create.

💬 *"I made my config a typed schema and wrote a test that pins the pre-registered values. If I ever change a primary parameter after seeing results, my own test suite stops me and makes me justify it."*

---

## 7. Phase-by-phase execution plan

> Original spec = 1 week. This plan keeps that spirit but front-loads correctness. Each phase is independently runnable and cached.

**Phase status at a glance** — the authoritative, always-current version of this table lives in **[STATUS.md](STATUS.md)**.

| Phase | Scope | State |
|---|---|---|
| **0** | Scaffolding, environment, typed config, test harness | ✅ **complete** — 2026-08-04 |
| **1** | FOMC statements + SPY prices + leak-free alignment → `panel.parquet` | ✅ **complete** — 2026-08-04 |
| **2** | FinBERT scoring → `sentiment_scores.csv` | ⬜ **next** |
| **3** | Trailing Z-score signal + positions | ⬜ |
| **4** | IC, bootstrap CIs, diagnostics, research notebook | ⬜ |
| **5** | Robustness sweeps, minutes corpus | ⬜ (stretch) |

---

### Phase 0 — Scaffolding (½ day) ✅ **COMPLETE** — 2026-08-04

**Goal:** a clean, reproducible skeleton before any real logic.

**Tasks — as executed:**
1. ✅ Directory tree per §6.1; `git init` on branch `main` in `C:\Users\Aryan\Work\QRT`.
2. ✅ 🔧 `.venv` on **Python 3.12.12** via **`uv`**, with `pyproject.toml` + `uv.lock` instead of a bare `requirements.txt` (rationale: §6.2.1). `requirements.txt` exported anyway.
3. ✅ 🔧 `torch` pinned to the **cu128** index for Blackwell `sm_120` (§6.2.2), verified by an **actual kernel launch**, not by `cuda.is_available()`.
4. ✅ `config.yaml` written with the pre-registered primary configuration (§5.1, Appendix A), **validated by a pydantic schema** (§6.3.1).
5. ✅ `README.md` written (setup, CUDA notes, layout, conventions).
6. ✅ `pytest` configured; **31 real tests passing**, not a placeholder.
7. ✅ 🔧 Leak canaries for Phases 1–3 **written and skipped**, so those phases are built *against* their invariants rather than having invariants retrofitted afterwards.

**✅ DoD — verified, with the commands that verified it:**

| Criterion | Command | Result |
|---|---|---|
| Repo builds | `uv sync` | exit 0 |
| All deps importable | `uv run pytest tests/test_environment.py` | 21 passed |
| `pytest` runs | `uv run pytest` | **31 passed, 15 skipped** |
| CUDA genuinely works | `uv run python scripts/check_gpu.py` | `PASS: kernel executed and returned a numerically correct result` |
| Lint clean | `uv run ruff check . && uv run ruff format --check .` | clean |
| FinBERT loads and runs | manual GPU smoke test | ✅ (and produced the two findings below) |

**Two findings that changed later phases:**
1. **FinBERT's label ordering is `{0: positive, 1: negative, 2: neutral}`** — not alphabetical, not the common convention. Indexing logits by position silently inverts the whole signal. Recorded in §2.3; pinned by a test.
2. **FinBERT scores an archetypal hawkish sentence as mildly positive** (§2.3) — an early, concrete instance of the §2.7 sign trap and the §8.3 out-of-domain question.

**Two corrections made to this document:** the `Adj Close` / adjusted-Open incompatibility (§3.4) and the `release_date + 1` fallacy (§3.3).

**Deliverable:** ✅ runnable, tested, GPU-verified repo committed to git.

---

### Phase 1 — Data pipeline (Days 1–2) ✅ **COMPLETE** — 2026-08-04

**Delivered:** `data/processed/panel.parquet`, **n = 225** events (2000-02-02 → 2026-07-29), 212 scheduled + 13 unscheduled, **51 tests passing** (up from 31), ruff clean.

**✅ DoD — verified, with the command that verified each item:**

| Criterion | Command | Result |
|---|---|---|
| NYSE calendar cross-check empty | `scripts/fetch_prices.py` | **PASS** — 8,434 sessions, **zero** discrepancies over 33.5 years |
| Statements scraped + hashed | `scripts/scrape_fomc.py` | 393 documents (244 statements, 149 minutes), **manifest integrity PASS** |
| Realised count printed, not assumed | `scripts/build_panel.py` | funnel printed: 393 → 244 → 241 → **225** |
| Zero silent drops | same | `244 == 241 + 3` ✅ (3 flagged, all 1994–95, all outside the sample) |
| 5 known meetings spot-checked | same | all correct, incl. the two pre-open cases |
| `test_align_no_lookahead` passes | `uv run pytest` | **20/20** (11 synthetic + 9 integration) |
| Forward-return NaNs only in the tail | `scripts/build_panel.py` | 0 NaN at *h*=1; exactly 1 at *h*≥3 (the most recent meeting) |
| Overlap verified, not assumed | same | **measured and reported** — see the correction in §4.3 |

**Five findings that changed the project:**

1. **The disclosure regime changed** (§3.1.1) — the Fed only announced policy *changes* until 1999; 1996 has zero statements. Sample start moved 1994 → 2000.
2. **Four URL eras, not three** — the `boarddocs` era (1997–2005, **61 statements**) was missing from the plan. A pattern-based scraper would have silently returned nothing for nine years.
3. **Documents state their own release time** (§3.3.1) — 88 statements now carry an evidence-based timestamp, and both hand-entered overrides were independently confirmed.
4. **Overlap was materially understated** (§4.3) — 16 overlapping pairs at *h*=20, minimum gap **3 sessions**, not the "~32" the average implied.
5. **A soft-404 body is ~1,170 characters** — above the 500-char flag threshold, so a length check alone is insufficient; content is checked too.

**Two bugs the discovery dry-run caught before any download** — neither would have raised an error:
- `_is_scheduled()` returned `False` for "no evidence", mislabelling **all 61 modern-era statements** as unscheduled.
- The `monetary{date}a.htm` suffix is *not* sufficient in the modern era: the annual *Statement on Longer-Run Goals* lives there too and was being ingested as a policy statement.

---

<details>
<summary><b>Phase 1 as planned</b> (kept for comparison with what actually happened)</summary>

**Goal:** a leak-free, timestamped, merged dataset of FOMC text + SPY prices. **No sentiment yet.**

🔧 **Build order changed: `prices.py` first, not the scraper.** Two reasons. (a) It is the smaller, independently verifiable piece, so it fails fast if `yfinance` is misbehaving. (b) Its output — the trading-day index — is the **input** `align.py` needs as its source of truth, so building it first means never having to stub it.

#### Task 1 — `prices.py`

`yfinance` download of SPY daily OHLCV with **`auto_adjust=True`** (§3.4), 1993 → today. Cache to `data/processed/spy_prices.parquet`. The price index becomes the trading-day calendar; `validate_against_nyse()` cross-checks it against `pandas_market_calendars` **in both directions** and must return an empty discrepancy frame.

⚠️ **`yfinance` is an unofficial scraper of an undocumented endpoint.** It can rate-limit, change response shape, or return partial history without raising. The parquet cache is therefore the working copy and the network is touched only when the cache is missing or a refresh is explicitly requested. A single missing session shifts an entry date by one and corrupts every downstream return — which is exactly why the NYSE cross-check exists.

#### Task 2 — `scrape_fomc.py`

🔧 **Do not construct document URLs. Crawl the index pages and harvest their anchors.** Verified in Phase 0, the statement URL convention has changed **three times**:

| Era | Pattern | Verified example |
|---|---|---|
| 1994–2005 | `/fomc/{YYYYMMDD}default.htm` | `/fomc/19940204default.htm` |
| 2006–2018 | `/newsevents/press/monetary/{YYYYMMDD}a.htm` | |
| 2019–now | `/newsevents/pressreleases/monetary{YYYYMMDD}a.htm` | |

The index pages have *also* split: `fomccalendars.htm` covers **2021 → present only**; everything older lives at `fomchistorical{YYYY}.htm` (confirmed: `fomchistorical1994.htm` exists and lists 10 meetings plus 4 conference calls).

A pattern-based scraper must encode three document conventions and will silently return 404s if a fourth appears. An index-crawling scraper encodes **two** index URLs and inherits whatever the Fed links to. **Index-page structure is far more stable than document-URL conventions**, so this is strictly more robust *and* less code.

🧠 Note the confirmation hidden in that example: the **February 3–4, 1994** meeting is filed under `19940204` — **day 2, the decision day**. That is exactly the `event_date` convention §3.2 assumes, verified at the very start of the sample rather than hoped for.

Extract per document: `event_date`, `release_datetime` (tz-aware, from the 14:00 ET default or a `manual_override`), `release_time_source`, `url`, `raw_path`, `text`, `n_chars`. Write `data/interim/documents.parquet`.

⚠️ **Be a polite scraper.** Set a descriptive `User-Agent` with contact details, sleep 1 s between requests, retry with exponential backoff, and **skip URLs already in the manifest** so the crawl is resumable and the site is hit exactly once per document, ever. ~270 documents at 1 req/s ≈ 5 minutes, once, forever. The archive is static; there is no excuse for a second pass.

#### Task 3 — `parse_statement()`: the highest-risk function in the phase

Three page-template eras means three different HTML structures, so selectors are brittle. Parse defensively: try a sequence of content selectors, fall back to whole-page text extraction rather than raising.

⚠️ **The critical requirement: a document that yields suspiciously little text must be FLAGGED, never silently passed through.** An empty or truncated statement scores as *neutral*, which is a perfectly plausible-looking number. It does not crash, it does not produce a NaN, it just quietly pulls the signal toward zero and dilutes the IC. **Silent degradation is more dangerous than loud failure**, because nothing prompts you to look.

#### Task 4 — `align.py`

Implement the alignment rule (§3.3) via `searchsorted(side="right")`. Compute forward returns for `h ∈ {1,3,5,10,20}`, open-to-open on adjusted prices. Run `check_overlap()` to **verify rather than assume** the ~32-trading-day event spacing that keeps *h* ≤ 20 windows non-overlapping (§4.3). Produce `data/processed/panel.parquet`.

#### Task 5 — un-skip `tests/test_align_no_lookahead.py`

Five assertions, already written:
1. `entry_date > release_datetime` **strictly**, for every row. Equality is a failure.
2. Every `entry_date` is a member of the trading-day index.
3. `entry_date` is the **first** such day — not merely *some* later one. (Skipping ahead would be conservative rather than leaky, but it would silently change the horizon being measured.)
4. The two emergency meetings align correctly: **2020-03-15** (Sunday 17:00 ET) → 2020-03-16; **2008-10-08** (07:00 ET, pre-open) → 2008-10-09.
5. Forward-return NaNs appear **only** in the unavoidable tail.

🧠 **Concepts reinforced:** immutability of raw data, provenance hashing, trading calendars, the release-vs-event timestamp distinction, timezone-aware timestamps, the "next-open" leak guard, and defensive parsing.

**✅ DoD:**
- `spy_prices.parquet` exists; `validate_against_nyse()` returns an empty discrepancy frame.
- ~270 statements in `data/raw/fomc_statements/`, each with a manifest row; `verify_manifest()` reports zero hash mismatches.
- **The realised document count is printed and recorded** (§3.5) — not assumed.
- Every document has both timestamps; a spot-check of 5 known meetings (2001-09-17, 2008-10-08, 2008-12-16, 2020-03-15, and one recent scheduled meeting) matches reality.
- Zero documents dropped silently: `n_scraped == n_parsed + n_flagged`, with the flagged list printed.
- `test_align_no_lookahead` passes — all 5 assertions.
- `panel.parquet` has no NaNs in forward-return columns except the unavoidable tail.
- `check_overlap()` confirms minimum event spacing > 20 trading days (or the exceptions are enumerated).

**Deliverable:** `panel.parquet` + passing alignment test.

</details>

---

### Phase 2 — NLP engine (Days 3–4)

**Goal:** turn each document into a raw sentiment score $S_t$, cached to CSV.

**Tasks:**
1. **`sentiment.py`** —
   - Load `ProsusAI/finbert` via `transformers` (`AutoTokenizer`, `AutoModelForSequenceClassification`), set `model.eval()`, run on CPU (fine for this scale) or GPU if available.
   - `split_sentences(text)` via `nltk.sent_tokenize` with regex fallback.
   - `score_sentence(sent) -> (p_pos, p_neg, p_neu, label)` — tokenize with `truncation=True, max_length=512`, softmax the logits. (Sentences are short, but truncation guards against pathological long "sentences.")
   - `aggregate(document) -> {S_count, S_prob, n_sentences, ...}` implementing **both** formulas from 4.1.
   - Batch sentences for speed; wrap in `torch.no_grad()`.
   - **Save every sentence-level score** too (to a side file) — we need the extreme sentences for the Section 8.3 confusion matrix, and we don't want to re-run FinBERT to get them.
2. Run across the full corpus **once**; write `data/processed/sentiment_scores.csv` (one row per document: both aggregation scores + metadata) and a `sentence_scores.parquet` (one row per sentence).
3. **`tests/test_sentiment_shapes.py`** — probabilities sum to ~1; scores in $[-1,1]$; no document lost.

🧠 **Concepts reinforced:** the 512-token limit and why we chunk; probability vs. count aggregation; **caching the expensive model output** so Phase 3–4 iterate instantly; FinBERT's calibration on out-of-domain (Fed) text as an open question.

⚠️ **Determinism:** set seeds and use `eval()`/`no_grad()` so re-runs are byte-identical — reproducibility is a research value *and* an interview talking point.

**✅ DoD:** `sentiment_scores.csv` exists with one row per parsed document (~270 expected — the exact count established in Phase 1, not assumed), both `S_count` and `S_prob` populated; sentence-level cache saved; shape tests pass, **including the `id2label` ordering assertion**; re-running produces byte-identical numbers.
**Deliverable:** the two cache files. **Commit** (but consider whether to commit large data — likely `.gitignore` the CSVs and commit a checksum + the code that regenerates them).

---

### Phase 3 — Alpha signal (Day 5)

**Goal:** convert raw sentiment into the tradable Z-score signal and positions.

**Tasks:**
1. **`alpha_signal.py`** (🔧 renamed from `signal.py`; see §6.1) —
   - Merge `sentiment_scores.csv` with `panel.parquet` on document.
   - Compute the **trailing** rolling Z-score with the mandatory `.shift(1)` (Section 4.2). Parameterize *L* from config.
   - Emit `position ∈ {-1, 0, +1}` from the threshold rule (4.4), sign taken from the pre-registered hypothesis.
2. **`tests/test_zscore_no_lookahead.py`** — reconstruct `Z_t` using only rows `< t` and assert it matches the vectorized column exactly. This is the **leak canary** for the whole project.

🧠 **Concepts reinforced:** surprises-not-levels, stationarity, the `.shift(1)` leak guard, empirical (not assumed) sign.

**✅ DoD:** signal column produced; **both** no-lookahead tests green; the first *L* meetings correctly have `NaN`/`0` signal (undefined window).
**Deliverable:** `panel.parquet` augmented with `S`, `Z`, `position`. **Commit.**

---

### Phase 4 — Backtest, IC & analytics (Days 6–7)

**Goal:** the honest evaluation and the research notebook.

**Tasks:**
1. **`backtest.py`** —
   - Vectorized strategy returns: `strat_ret = position * fwd_ret_h` for the primary horizon; subtract transaction costs (Section 8.2).
   - **Information Coefficient:** `scipy.stats.spearmanr(Z, fwd_ret_h)` (primary) and Pearson (secondary), per horizon.
   - **IC t-stat** and **bootstrap 95% CI** (Section 5.2).
2. **`diagnostics.py`** — the three "QRT Touch" plots (Section 8) + the full IC grid table.
3. **Tearsheet:** feed the strategy return series to `quantstats.reports.html(...)` for Sharpe, MDD, win rate, etc. (Fallback: compute the headline metrics by hand if quantstats/pandas versions clash.)
4. **`notebooks/99_research_report.ipynb`** — the narrative: hypothesis → method → leak guards → IC with CI → decay → cost sensitivity → Fed-speak confusion matrix → **honest conclusion**.

🧠 **Concepts reinforced:** vectorized backtesting, IC as primary metric, error bars over point estimates, cost realism, qualitative model validation.

**✅ DoD:**
- IC reported **with** t-stat and bootstrap CI for every horizon.
- All three diagnostic plots render.
- Tearsheet generated (or hand-rolled metrics).
- Notebook reads as a coherent story and states a **falsifiable, honestly-caveated conclusion** — positive *or* negative.

**Deliverable:** the rendered notebook + tearsheet HTML. **Commit + tag `v1.0`.**

---

### Phase 5 (stretch, optional) — Robustness & extensions

Only after v1.0 is honest and complete:
- Add the **minutes** signal; compare/combine with statements.
- Robustness sweep over *L* ∈ {6,12}, *θ* ∈ {0.5,1.0,1.5}, aggregation ∈ {count,prob}, entry ∈ {next-open, same-close} — presented as a **grid**, framed as robustness not cherry-picking.
- Boilerplate stripping; per-section scoring (e.g., score only the forward-guidance paragraph).
- Walk-forward evaluation (Section 5.3) if sample permits.

---

## 8. The "QRT Touch": research-grade diagnostics

These three artifacts are the difference between "a backtest" and "research." Each has a **why**.

### 8.1 IC Decay bar chart

**What:** bar chart of Spearman IC at *h* = 1, 3, 5, 10, 20 trading days, each with its bootstrap CI as an error bar.
**Why it impresses:** it proves you know **alpha has a half-life** and that you think about *when* the market prices information in. A monotone decay is the economically sensible shape; a flat line invites the question "are your returns overlapping?" (which you'll have already checked).
💬 *"The decay tells me the tradability window. If the IC lives at h=1 but dies by h=5, transaction costs will likely eat it — which is exactly what plot #2 tests."*

### 8.2 Transaction-cost sensitivity curve

**What:** line chart of strategy **Sharpe ratio** as per-trade cost rises 0 → 1 → 2 → 5 bps (and beyond). Cost model: subtract `cost_bps × |Δposition|` each event (you pay when you enter/flip/exit).
**Why it impresses:** **most student backtests ignore costs and are therefore fiction.** Showing the Sharpe *degrade* with realistic frictions proves you know a paper edge and a real edge are different animals. Even a few bps can flip a marginal strategy negative — and you *want* to show whether yours survives.
💬 *"A signal is only alpha if it survives the cost of harvesting it. This curve is my go/no-go test for real-world viability."*

### 8.3 The "Fed-speak confusion matrix" (qualitative model audit)

**What:** two things —
1. Extract, from the cached sentence scores, the **top-10 most negative (hawkish-flavored)** and **top-10 most positive (dovish-flavored)** sentences FinBERT found. Read them like a human. Do they *look* hawkish/dovish? This is a **sanity check on the black box.**
2. Optionally, a genuine **confusion matrix**: on a small hand-labeled sample of Fed sentences (you label ~30 as hawk/dove/neutral), compare to FinBERT's labels with `sklearn.metrics.confusion_matrix`. Quantifies FinBERT's agreement with domain-expert judgment on *central-bank* language specifically.
**Why it impresses:** it proves you **did not blindly trust the AI.** You interrogated whether a news-trained model actually comprehends Fed-speak — the exact skepticism a research desk wants. If FinBERT systematically mislabels hedged Fed language, *that's a finding*, and it explains a weak IC.
💬 *"FinBERT was trained on analyst news, not central-bank prose. Before trusting its signal, I audited its most extreme classifications by hand and against a small labeled set — model interpretability isn't optional when the model is upstream of every trade."*

---

## 9. Interview narrative

Have a crisp 90-second story ready. Structure:

1. **Hypothesis (economics first):** "Markets price *surprises* in Fed tone. I tested whether the *change* in FOMC-statement sentiment predicts short-horizon SPY returns."
2. **Leak discipline:** "The whole pipeline is built chronologically to prevent look-ahead bias — statements are stamped with their 2 PM release, minutes with their true 3-weeks-later release, and every signal trades at the *next* open. I unit-test that no signal uses future data."
3. **Method:** "FinBERT scores each sentence; I aggregate to a document score, then take a trailing Z-score so I'm measuring surprise, not level."
4. **Honest evaluation:** "Primary metric is Spearman IC with a bootstrap confidence interval — because with only ~263 events, the point estimate is meaningless without error bars. I report the full horizon grid, not the best cell."
5. **Power, stated up front:** "I ran the power analysis before the backtest. At n≈263 the minimum detectable IC is about 0.12, and a genuinely institution-grade signal is 0.05 — so this design is underpowered by an order of magnitude, and the Fundamental Law gives the same answer from the economics side: 8 events a year caps the information ratio around 0.3 regardless of how good the signal is. The fix is breadth, not more history."
6. **Research maturity:** "I plotted IC decay, stress-tested against transaction costs, and audited FinBERT's understanding of Fed-speak by hand — it scores an inflation-vigilance sentence as *positive*, which is exactly the sentiment-versus-equity-direction confusion I'd flagged as a risk. My conclusion is [X], with this confidence interval."

**Questions they may ask — and your answers:**
- *"How do you know you didn't overfit?"* → pre-registered primary config; reported the whole grid; tiny trial count; bootstrap CIs; (optional) walk-forward.
- *"Why Spearman not Pearson?"* → robustness to fat tails/outliers in returns.
- *"Your Sharpe is 4 — believe it?"* → No; at n≈263 that's almost certainly overfit or a leak; the IC t-stat and CI are the trustworthy numbers. The Fundamental Law caps my plausible IR near 0.3 at 8 events/year, so a Sharpe of 4 would contradict the arithmetic of my own design.
- 🔧 *"Your result is null — so what did you learn?"* → That the design is underpowered by an order of magnitude, which I knew in advance and quantified: MDE ≈ 0.12 against a strong-signal benchmark of 0.05. The informative output is the power analysis and the breadth argument, not the IC point estimate.
- 🔧 *"How do you know FinBERT understands Fed language?"* → I don't assume it. It scores "the Committee remains highly attentive to inflation risks" as 34% positive — a hawkish statement read as positive sentiment. That's the sentiment-vs-equity-direction gap, and §8.3 quantifies it against hand labels rather than hand-waving it.
- *"Positive sentiment → stocks up?"* → not assumed; hawkish 'positive' can be bearish; I set the sign from a dovish-surprise-is-bullish hypothesis and tested it, rather than fitting the sign to returns.
- *"What would you do with more time?"* → add minutes & press-conference transcripts, model regime-dependence (the Fed→market relationship differs in ZIRP vs. hiking cycles), and event-study intraday windows.

---

## 10. Risk register & known limitations

Status column: 🛡️ = mitigation implemented and verified; 🔶 = mitigation designed, not yet built; ⬜ = open, accepted risk.

| Risk / limitation | Impact | Mitigation | Status |
|---|---|---|---|
| **Tiny sample (n≈263)** — minimum detectable IC ≈ 0.12 vs. a strong signal of 0.05 | **The dominant limitation.** Low power; wide CIs | Report CIs & t-stats; power analysis (§14.5); frame a negative as a valid, primary result | 🔶 Phase 4 |
| **Look-ahead via minutes lag** | Fatal (fake alpha) | Separate release-date scraping; `release_datetime` is the only tradable timestamp | 🔶 Phase 5 |
| **Look-ahead via Z-score window** | Fatal | Mandatory `.shift(1)`; `test_zscore_no_lookahead` rebuilds Z from rows `< t` independently | 🔶 written, skipped |
| **Look-ahead via entry date** | Fatal | `searchsorted(side="right")`; `test_align_no_lookahead` (5 assertions incl. the two emergency meetings) | 🔶 written, skipped |
| 🔧 **FinBERT label ordering is `{0:pos, 1:neg, 2:neu}`** | **Silent total sign inversion** — nothing crashes | Index by name from `config.id2label`, never by position; pinned by a test | 🛡️ documented §2.3 |
| **FinBERT out-of-domain on Fed-speak** | Weak/biased signal | Confusion-matrix audit (§8.3); count vs prob aggregation; **already observed** scoring a hawkish sentence as positive | 🔶 Phase 4 |
| **Overfitting the knobs** | Illusory edge | Pre-registration in `config.yaml`, **enforced by `test_preregistered_primary_values`**; full-grid reporting | 🛡️ §6.3.1 |
| **Ignoring costs** | Fictional Sharpe | Cost-sensitivity curve (§8.2); `bps_grid: [0,1,2,5,10]` in config | 🔶 Phase 4 |
| 🔧 **Adjusted-price / dividend errors** | Biased returns (~1.3%/yr) | `auto_adjust: true` on all four OHLC columns; open-to-open; `test_price_fields_are_consistent` | 🛡️ §3.4 |
| **Silent parse failure → phantom neutral scores** | Dilutes signal invisibly | Flags short extractions **and soft-404 bodies** (a Fed 404 renders ~1,170 chars, above the length threshold — length alone is insufficient); `n_scraped == n_parsed + n_flagged` asserted | 🛡️ §7 Phase 1 |
| 🔧 **Overlapping forward returns at long horizons** | Inflated t-stats; unearned significance | **Measured, not assumed:** 16 pairs at *h*=20, min gap 3 sessions. Headline moves to *h* ∈ {1,3,5}; Newey–West at *h* ≥ 10; `include_unscheduled: false` robustness run | 🛡️ measured §4.3 |
| 🔧 **Disclosure-regime break (1994–1998)** | Selection effect + a 28-month Z-window posing as 9 months | Sample starts 2000-01-01; pre-2000 documents retained as a labelled robustness check | 🛡️ §3.1.1 |
| 🔧 **Unverified release times** | Guessing early = look-ahead | Times parsed from the documents (88/225); config schema **enforces** the default is post-open; precedence keeps a regex from overturning researched values | 🛡️ §3.3.1 |
| **Regime dependence** (Fed↔market link varies) | Non-stationary edge | Discuss; optional regime split in Phase 5 | ⬜ accepted |
| **`quantstats`/pandas version clash** | Broken tearsheet | Moved behind an optional `--extra tearsheet`; headline metrics hand-computed so results never depend on it | 🛡️ §6.2.4 |
| **Scraper breakage / site format drift** | Pipeline stalls | Crawl index pages, never construct document URLs; cache raw once; sha256 manifest; parse defensively | 🔶 Phase 1 |
| 🔧 **`yfinance` is an unofficial API** | Silent partial history | Parquet cache is the working copy; `validate_against_nyse()` cross-checks both directions | 🔶 Phase 1 |
| 🔧 **`transformers` 5.x, ahead of the planned 4.x** | 4.x tutorial code may not transfer | Verified FinBERT loads and runs on 5.14.1; lockfile freezes the version | 🛡️ §6.2.4 |
| 🔧 **CUDA wheel without `sm_120` kernels** | Passes every check, then fails at first matmul | torch pinned to the cu128 index; verified by a **real kernel launch**, not `is_available()` | 🛡️ §6.2.2 |
| 🔧 **Documentation drifting from the code** | You build on a false model of your own work | The §0.2 maintenance protocol; STATUS.md; the §16 revision log | 🛡️ this session |

---

## 11. Glossary quick-reference

**Quant & statistics**
- **Look-ahead bias** — using future info in a past decision. The cardinal sin.
- **IC (Information Coefficient)** — corr(signal_t, return_{t→t+h}); Spearman = rank-based. 0.05 is strong.
- **IC decay** — how IC falls as horizon grows; reveals alpha half-life.
- **Alpha / signal / factor** — predictable excess return / the number that predicts it / a cross-asset driver of returns.
- **Hawkish / Dovish** — tighter (bearish for stocks) / easier (bullish) policy lean.
- **Sharpe ratio** — excess return ÷ volatility. >3 in a student backtest ⇒ suspect.
- **Max drawdown (MDD)** — largest peak-to-trough loss.
- **bp (basis point)** — 0.01% = 0.0001.
- **Z-score** — (x − trailing mean) ÷ trailing std; measures surprise, not level.
- **Purging / embargo** — removing overlapping-label samples across a train/test split.
- **Bootstrap CI** — resampling-based, assumption-free confidence interval.
- **Turnover** — how much you trade; drives cost.
- **OHLC / adjusted prices** — price bar fields; adjusted = dividend/split-corrected.
- 🔧 **ETF** — a fund whose *shares* trade on an exchange like a stock. SPY holds the S&P 500.
- 🔧 **Panel** — a table with one row per observation and one column per variable.
- 🔧 **Statistical power** — the probability of detecting an effect that is genuinely there. Low power ⇒ a true signal reads as "not significant." (§14.5)
- 🔧 **Minimum detectable effect (MDE)** — the smallest true effect a sample size *could* resolve. Ours is IC ≈ 0.12 at *n* = 263.
- 🔧 **Standard error** — the standard deviation of an *estimate* across hypothetical repeated samples. Shrinks like 1/√n.
- 🔧 **Fundamental Law of Active Management** — IR ≈ IC × √breadth. Explains why 8 events/year is a hard ceiling. (§14.6)

**NLP & modelling**
- **FinBERT** — BERT fine-tuned on financial text; outputs P(pos/neg/neu).
- 🔧 **Token** — a sub-word unit produced by a tokenizer. BERT's positional table holds exactly **512**.
- 🔧 **Logits** — a model's raw, unnormalised output scores, before softmax.
- 🔧 **Softmax** — the map from logits to a probability distribution: $p_i = e^{z_i}/\sum_j e^{z_j}$.
- 🔧 **Calibration** — whether a stated 70% confidence is right ~70% of the time. FinBERT's calibration on Fed prose is unknown and is precisely what §8.3 probes.
- 🔧 **`id2label`** — a checkpoint's class-index → class-name map. FinBERT's is `{0:positive, 1:negative, 2:neutral}` — index by **name**, never position.
- 🔧 **Out-of-domain** — applying a model to text unlike its training distribution. FinBERT was trained on analyst news, not central-bank prose.

**Engineering**
- 🔧 **Compute capability (`sm_XX`)** — a GPU's instruction-set version. RTX 5070 Ti = `sm_120`.
- 🔧 **cubin / PTX** — pre-compiled kernels for one `sm_XX` / forward-compatible intermediate assembly the driver can JIT.
- 🔧 **Lockfile** — pins exact versions *and hashes* of every transitive dependency. What makes "reproducible" true.
- 🔧 **`searchsorted`** — binary search returning an insertion index; with `side="right"`, the first element strictly greater than the key. Our entire alignment rule.
- 🔧 **sha256** — a cryptographic fingerprint of bytes; makes "raw data is immutable" verifiable rather than aspirational.
- 🔧 **tz-aware timestamp** — a datetime carrying its timezone. Naive datetimes silently shift days when they touch UTC.
- 🔧 **Module shadowing** — a local `signal.py` masking the stdlib `signal`, breaking any library that imports it (e.g. torch).

---

## 12. References

- Marcos López de Prado, *Advances in Financial Machine Learning* (2018) — purging, embargo, deflated Sharpe, backtest overfitting. **The** reference for this project's rigor.
- Araci (2019), *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models* — the model family. `ProsusAI/finbert` on Hugging Face.
- Grinold & Kahn, *Active Portfolio Management* — the "Fundamental Law of Active Management" and IC as the core measure of skill.
- Federal Reserve, FOMC materials archive (federalreserve.gov/monetarypolicy/fomc_historical.htm) — primary data source; note release-lag rules for minutes.
- Bailey & López de Prado, *The Deflated Sharpe Ratio* (2014) — correcting Sharpe for multiple testing.

---

### Appendix A — The pre-registered primary configuration

✅ **Written and validated.** The authoritative version is [`config.yaml`](config.yaml) in the repository root — heavily commented, parsed by a pydantic schema (§6.3.1), and pinned by `tests/test_config.py::test_preregistered_primary_values`. The block below is the abridged form; **if the two ever disagree, `config.yaml` is correct and this appendix is stale.**

```yaml
# Pre-registered PRIMARY configuration (decided before viewing any returns).
data:
  start_date: "1994-01-01"       # first regular FOMC statements
  ticker: "SPY"
  auto_adjust: true              # 🔧 CORRECTED: adjusts ALL FOUR OHLC columns
  entry_price_field: "Open"      # 🔧 was `price_field: "Adj Close"` — see §3.4
  exit_price_field: "Open"       #    both legs on the same basis
sentiment:
  model: "ProsusAI/finbert"
  aggregation: "prob"            # primary = probability-weighted (S_prob)
  max_length: 512                # BERT's positional-embedding limit
  device: "auto"                 # 🔧 cuda if available, else cpu — portable
  seed: 42
  strip_boilerplate: false       # primary = OFF; stripping is a robustness check
signal:
  zscore_window_L: 6             # trailing meetings, EXCLUDES current (.shift(1))
  threshold_theta: 1.0
  hypothesis_sign: "dovish_surprise_bullish"   # a HYPOTHESIS, tested — not fitted
backtest:
  primary_horizon_days: 5
  horizons_days: [1, 3, 5, 10, 20]
  entry: "next_trading_open"     # strict leak guard
costs:
  bps_grid: [0, 1, 2, 5, 10]     # 🔧 added 10 bp — a realistic retail-ish upper bound
evaluation:
  ic_method_primary: "spearman"
  bootstrap_iters: 10000
  bootstrap_ci: 0.95
  random_seed: 42
```

Plus a `scrape:` block (index URLs, politeness settings, the 14:00 ET default and the two `release_time_overrides`) and a `robustness:` block enumerating the sweeps that are **explicitly not** candidates for the headline result.

*Everything not in the primary block is a robustness check, not a competing candidate for "the result."*

---

## 13. Component reference

*Every module: what it does, how you use it, what it depends on. If you cannot answer those three questions about a unit of code, its boundaries are wrong.*

### 13.1 The dependency graph

Data flows strictly one way. Nothing downstream is imported upstream, so any stage can be rebuilt without touching the ones before it.

```
config.yaml ──▶ src/config.py ──┐   (every module takes a validated Config)
                                │
  federalreserve.gov ──▶ scrape_fomc.py ──▶ data/raw/*.htm + manifest.csv
                                │                     │
                                │                     ▼
                                │           data/interim/documents.parquet
                                │                     │
   Yahoo Finance ──▶ prices.py ─┤                     │
                                ▼                     ▼
                    spy_prices.parquet ────▶ align.py ────▶ panel.parquet
                                                                  │
                     documents.parquet ──▶ sentiment.py           │
                                                │                 │
                          sentiment_scores.csv ─┤                 │
                          sentence_scores.parquet                 │
                                                ▼                 ▼
                                        alpha_signal.py ◀─────────┘
                                                │
                                    panel.parquet (+ S, Z, position)
                                                │
                                   ┌────────────┴────────────┐
                                   ▼                         ▼
                             backtest.py              diagnostics.py
                                   │                         │
                                   └──────▶ notebooks/99_research_report.ipynb
                                                  + reports/*.html
```

### 13.2 Module contracts

| Module | Purpose (one line) | Key functions | Depends on | Produces |
|---|---|---|---|---|
| **`config.py`** ✅ | Turn `config.yaml` into a validated, immutable object | `load_config()`, `Config`, `PathsConfig.resolve()` | `pyyaml`, `pydantic` | a `Config` — nothing on disk |
| **`scrape_fomc.py`** | Acquire FOMC text with honest timestamps and provenance | `discover_documents()`, `download_documents()`, `parse_statement()`, `build_documents_table()`, `verify_manifest()` | `requests`, `bs4`, `lxml` | `data/raw/**`, `manifest.csv`, `documents.parquet` |
| **`prices.py`** | Adjusted SPY bars + the authoritative trading calendar | `download_prices()`, `trading_days()`, `validate_against_nyse()` | `yfinance`, `pandas_market_calendars` | `spy_prices.parquet` |
| **`align.py`** | **The leak guard.** Release → tradable entry → forward returns | `next_trading_open()`, `forward_returns()`, `check_overlap()`, `build_panel()` | `prices`, `scrape_fomc` | `panel.parquet` |
| **`sentiment.py`** | Text → numbers via FinBERT, cached | `resolve_device()`, `split_sentences()`, `score_sentences()`, `aggregate()`, `strip_boilerplate()`, `run_corpus()` | `torch`, `transformers`, `nltk` | `sentiment_scores.csv`, `sentence_scores.parquet` |
| **`alpha_signal.py`** | Raw score → surprise → position | `rolling_zscore()`, `positions_from_z()`, `build_signal()` | `sentiment`, `align` | `panel.parquet` + `S`, `Z`, `position` |
| **`backtest.py`** | The honest evaluation | `information_coefficient()`, `bootstrap_ic_ci()`, `ic_grid()`, `strategy_returns()`, `performance_metrics()` | `scipy`, `numpy` | result tables |
| **`diagnostics.py`** | The three research artifacts (§8) | `plot_ic_decay()`, `plot_cost_sensitivity()`, `extreme_sentences()`, `fedspeak_confusion_matrix()` | `matplotlib`, `seaborn`, `sklearn` | `reports/*.png` |

### 13.3 Design invariants that hold across every module

1. **`Config` in, artifact out.** Every top-level function takes a validated `Config`; no module reads `config.yaml` itself, and no magic numbers are inlined.
2. **Every stage is independently runnable and cached.** You can rebuild Phase 3 without re-scraping or re-scoring. This is what makes iteration cheap enough to actually do research in.
3. **Every transformation states what timestamp its inputs are valid as-of.** §2.1's discipline, applied as a comment convention.
4. **Failures are loud; degradation is flagged.** Nothing is silently dropped, defaulted, or coerced. A document that will not parse is *counted and reported*, never scored as neutral.
5. **The expensive things are hashed or cached.** The scrape is hashed (immutability is verifiable); the model output is cached (iteration is instant).

### 13.4 The test suite as a specification

The tests are not an afterthought — three of them were written **before** the code they test, so the implementation is developed against a fixed contract rather than the contract being retrofitted to whatever the code happens to do.

| Test file | Phase | What it actually guarantees |
|---|---|---|
| `test_environment.py` ✅ | 0 | 18 dependencies import; Python is 3.12; the torch build targets `sm_120`; **a real CUDA kernel launches and returns a numerically correct result** |
| `test_config.py` ✅ | 0 | Config parses; **primary values are pinned**; typos raise; `auto_adjust`/`Open` consistency; the two emergency-meeting overrides survive YAML round-tripping |
| `test_align_no_lookahead.py` 🔶 | 1 | `entry_date > release_datetime` strictly; entry is a real trading day; entry is the *first* such day; the 2020-03-15 and 2008-10-08 cases; NaNs only in the tail |
| `test_sentiment_shapes.py` 🔶 | 2 | `id2label` ordering; probabilities sum to 1; scores in [−1, 1]; no document lost; the regex fallback works; scoring is deterministic |
| `test_zscore_no_lookahead.py` 🔶 | 3 | **The canary.** Z rebuilt by a naive backward loop matches the vectorised column exactly; first *L* are NaN; **perturbing the future cannot change the past** |

🧠 **Why `test_zscore_no_lookahead` is written as an independent reimplementation.** If the test computed Z with `.rolling().shift()` — the same call the implementation uses — it could not possibly detect a missing `.shift(1)`, because both would be wrong identically. A test must fail for the bug it is named after. The loop version is slow and ugly on purpose: it shares no code with the thing it audits.

---

## 14. Methodology deep-dive

*§4 gives the formulas. This section derives them, and explains what each one assumes.*

### 14.1 Text → number: what FinBERT actually computes

A **transformer** maps a sequence of tokens to a sequence of context-aware vectors. BERT is the encoder-only variety: **bidirectional**, meaning each token's representation attends to tokens on both its left and right simultaneously. (Contrast a GPT-style decoder, which sees only leftward context because it is trained to predict the next token.)

The classification path is:

$$\text{sentence} \xrightarrow{\text{tokenizer}} [t_1 \dots t_n] \xrightarrow{\text{BERT}} \mathbf{h}_{\texttt{[CLS]}} \in \mathbb{R}^{768} \xrightarrow{W\mathbf{h}+b} \mathbf{z} \in \mathbb{R}^{3} \xrightarrow{\text{softmax}} (p^+, p^-, p^0)$$

- **`[CLS]`** is a special token prepended to every input; its final hidden state is conventionally used as a summary of the whole sequence.
- **Logits** $\mathbf{z}$ are raw, unnormalised scores — any real numbers.
- **Softmax** maps them to a probability distribution: $p_i = e^{z_i} / \sum_j e^{z_j}$. It is monotone (bigger logit ⇒ bigger probability) and the outputs sum to exactly 1, which is why `test_sentiment_shapes` can assert that as an invariant.

**Why 512 tokens is a hard wall.** BERT adds a learned **positional embedding** to each token so the model knows word order. That table has exactly 512 rows. Position 513 does not exist — this is not a performance limit but an architectural one, which is why documents must be split rather than truncated.

⚠️ **Softmax probabilities are not calibrated by default.** A **calibrated** model is one whose stated 70% confidence is correct ~70% of the time. Neural classifiers are typically over-confident, and *out-of-domain* — which Fed prose is, relative to FinBERT's analyst-news training data — calibration degrades further and unpredictably. This is the deep reason §4.1 computes **both** aggregations: $S^{\text{prob}}$ uses the probability magnitudes and is therefore exposed to miscalibration, while $S^{\text{count}}$ uses only the arg-max and is exposed only to *ranking* errors. Comparing them is a genuine experiment about the model, not a tie-break.

### 14.2 Why a Z-score, derived

Suppose the observed sentiment decomposes into a slow-moving baseline plus an innovation:

$$S_t = \mu_t + \varepsilon_t$$

where $\mu_t$ is the Fed's prevailing tone (which drifts across regimes — ZIRP, tightening cycles, crisis response) and $\varepsilon_t$ is this meeting's *departure* from it.

Markets, being forward-looking, have largely priced $\mu_t$ already: everyone knows the Fed's current stance. What can move prices is $\varepsilon_t$ — the part nobody knew. So the tradable quantity is the innovation, not the level, and we need an estimate of $\mu_t$ built **only from the past**:

$$\hat{\mu}_t = \frac{1}{L}\sum_{k=1}^{L} S_{t-k}, \qquad \hat{\sigma}_t = \text{sd}(S_{t-L}, \dots, S_{t-1}), \qquad Z_t = \frac{S_t - \hat{\mu}_t}{\hat{\sigma}_t}$$

Dividing by $\hat{\sigma}_t$ does two further jobs: it makes the signal **unit-free**, so a threshold like "±1 standard deviation" means the same thing in 2008 and 2021; and it *adapts* — in a period when the Fed's language is volatile, a given raw move counts for less.

**Choosing *L* is a bias–variance trade-off:**

| | small *L* (e.g. 3) | large *L* (e.g. 12) |
|---|---|---|
| Tracks regime shifts | quickly ✅ | slowly ❌ |
| Estimate of $\mu, \sigma$ | noisy ❌ | stable ✅ |
| Observations lost to the warm-up | few ✅ | many ❌ |

*L* = 6 (≈ 9 months of policy, three-quarters of a meeting cycle) is the pre-registered primary; *L* = 12 is a robustness check. **Both are declared in advance**, so this is a sensitivity analysis and not a search.

⚠️ **The `.shift(1)` in one sentence:** without it, $S_t$ appears in its own mean and standard deviation, so $Z_t$ is partly explained by a quantity that includes $Z_t$ — the estimate is contaminated by the very observation it is supposed to be surprised by, and the whole series becomes untradeable.

### 14.3 Why Spearman, not Pearson

**Pearson** correlation measures *linear* association and is a function of the actual magnitudes:

$$r = \frac{\sum (x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum (x_i-\bar x)^2}\sqrt{\sum (y_i-\bar y)^2}}$$

**Spearman** is Pearson applied to the **ranks** of the data. It asks only: *did a higher signal tend to go with a higher return?*

Financial returns have **fat tails** — extreme moves are far more common than a normal distribution predicts. With *n* ≈ 263 and a couple of ±5% FOMC days, a single observation can dominate the numerator of Pearson's formula. Your headline statistic then describes one day, dressed as a description of thirty years.

Spearman caps any observation's influence at its rank, so the largest possible contribution from one point is bounded. We report Pearson as a **secondary cross-check**: if the two disagree sharply, that is itself informative — it means the relationship is being driven by a few extreme events, which is worth knowing and worth saying.

🧠 **What we give up.** Spearman is invariant to any monotone transformation, so it cannot tell you *how big* the effect is, only how consistent. That is an acceptable trade here, because the honest deliverable (§5.5) is about whether an effect exists at all, not about sizing a book.

### 14.4 The IC t-statistic, derived

Under the null hypothesis that the true correlation $\rho = 0$, the sample correlation $r$ from *n* independent pairs satisfies:

$$t = r\sqrt{\frac{n-2}{1-r^2}} \sim t_{n-2}$$

Worked example at our sample size, *n* = 263:

| IC | *t* | Significant at 5% (|t| > 1.97)? |
|---|---|---|
| 0.05 (institution-grade) | 0.81 | ❌ no |
| 0.08 | 1.30 | ❌ no |
| 0.10 | 1.62 | ❌ no |
| **0.12** | **1.95** | **borderline** |
| 0.15 | 2.45 | ✅ yes |

⚠️ **Read that table again.** A signal at the level professionals consider excellent (0.05) produces *t* = 0.81 here — indistinguishable from noise. The three assumptions behind the formula are also worth naming, since two of them are shaky:
- **Independence.** Reasonable: events are ~32 trading days apart and horizons are ≤ 20, so windows do not overlap. `check_overlap()` verifies rather than assumes this.
- **Bivariate normality.** *Violated* — returns are fat-tailed. This is why the bootstrap CI, which assumes nothing about the distribution, is the more defensible error bar.
- **A single pre-specified test.** Violated the moment you look at a grid of horizons — which is why §5.1 reports the whole grid rather than its maximum.

### 14.5 🔧 Power analysis — the most important number in the project

**Definition — statistical power:** the probability that a test *detects* an effect, given that the effect is really there. Its complement is a **Type II error** (a false negative) — concluding "no edge" when there is one.

The standard error of a correlation near zero is approximately $\text{SE} \approx 1/\sqrt{n}$. At *n* = 263, SE ≈ 0.062, so the **minimum detectable effect** at the conventional 5% level is:

$$\text{MDE} \approx 1.96 \times 0.062 \approx \mathbf{0.12}$$

To reach 80% power (the usual convention) you need roughly $(1.96 + 0.84) \times \text{SE}$, i.e. an IC near **0.17**.

Inverting the question — how many events would be needed to detect a *realistic* signal at 80% power?

| True IC | Events needed | Years of FOMC meetings |
|---|---|---|
| 0.15 | ~350 | ~44 |
| **0.05** (institution-grade) | **~3,100** | **~390** |
| 0.03 | ~8,700 | ~1,090 |

**We would need four centuries of FOMC meetings to reliably detect a professionally excellent signal.** This is not a defect to apologise for — it is a *result*, and arguably the most valuable one the project produces, because it is true independently of what FinBERT does or how the backtest turns out.

It also tells you what to do about it, which is the difference between an excuse and an analysis: **increase breadth, not history.** Score many assets per event (rate futures, the dollar, sector ETFs, individual equities' sensitivity to rates), or move to intraday windows around the release. Both multiply the number of (signal, return) pairs per meeting rather than waiting for more meetings.

💬 *"Before running anything I computed the minimum detectable IC for my sample: about 0.12. A genuinely strong signal is 0.05. So I knew going in that a null result would be uninformative about the hypothesis and informative about the design — and I can tell you exactly what breadth I'd need instead."*

### 14.6 The Fundamental Law of Active Management

Grinold & Kahn's approximation:

$$\text{IR} \approx \text{IC} \times \sqrt{\text{breadth}}$$

where **IR** (Information Ratio) is risk-adjusted excess return — essentially a Sharpe ratio measured against a benchmark — and **breadth** is the number of *independent* bets per year.

Our breadth is **8** (one statement per meeting). So even a *very* strong IC of 0.10 gives:

$$\text{IR} \approx 0.10 \times \sqrt{8} \approx 0.28$$

An IR of 0.28 is not a fundable strategy. This is a second, independent route to the same conclusion as §14.5, arriving from economics rather than statistics: **the binding constraint is breadth, not signal quality.** Two different frameworks agreeing is much stronger evidence than either alone, and saying so is exactly the kind of reasoning a research desk wants to hear.

### 14.7 The bootstrap, and why it is the right error bar here

**Procedure:** from the *n* observed (signal, return) **pairs**, draw *n* pairs **with replacement**, recompute the IC, and repeat 10,000 times. The 2.5th and 97.5th percentiles of that distribution form the 95% confidence interval.

**Why it works.** The variation across resamples approximates the variation you would see across repeated draws from the population — so it estimates the sampling distribution of the statistic *empirically*, without needing a formula for it.

**Why it beats the t-statistic here:** it assumes no normality (ours is violated), it needs no closed form (there isn't a good one for Spearman under fat tails), and it naturally produces an *interval* rather than a binary verdict — which is exactly the honest deliverable §5.5 demands.

⚠️ **The one thing you must not get wrong: resample the PAIRS.** Resampling the signal and the returns *independently* destroys the association you are trying to measure and will hand you a confidence interval centred on zero, every time, for any data.

### 14.8 The cost model, and why it is not decoration

Cost per event is `cost_bps × |position_t − position_{t−1}|`. The absolute difference is the **turnover**: a flip from −1 to +1 costs twice a move from 0 to +1, because you must close one position and open the opposite.

The `bps` figure bundles several real frictions:

| Component | Typical, SPY | What it is |
|---|---|---|
| Bid–ask spread (half) | ~0.5 bp | SPY is one of the most liquid instruments on earth |
| Commission | ~0–0.5 bp | often zero retail |
| Slippage | 1–5 bp | the gap between your decision price and your fill |
| Market impact | ~0 at small size | your own trade moving the price |

Our grid `[0, 1, 2, 5, 10]` spans "frictionless fantasy" to "realistically pessimistic." The point is not to pick a number — it is to show **where the strategy dies**, and to state that number out loud.

🧠 **Why costs deserve their own plot rather than a single subtraction.** A single after-cost Sharpe hides the shape of the relationship. The curve answers the question an allocator actually asks: *how much execution slippage can this tolerate before it stops being worth doing?* A strategy that survives 10 bp is robust; one that dies at 1 bp was never real, and the difference is invisible if you only report one number.

### 14.9 How the pieces compose

Reading the pipeline as one sentence, with each step's assumption attached:

> **Fed language** *(assumes tone is measurable)* → **FinBERT sentence probabilities** *(assumes news-domain sentiment transfers to central-bank prose — §8.3 tests this)* → **document score $S_t$** *(assumes sentence scores aggregate meaningfully)* → **trailing Z-score $Z_t$** *(assumes markets price surprises, not levels)* → **position** *(assumes a dovish surprise is bullish — a hypothesis, tested via the sign of the IC)* → **forward return** *(assumes next-open entry is executable)* → **IC with a bootstrap CI** *(assumes events are independent — verified by `check_overlap`)*.

Every arrow is an assumption, and every assumption has either a test or an explicit acknowledgement. **That chain — not the final number — is the deliverable.** If someone attacks the result, they must attack a specific link, and for each link you can say what you did about it.

---

## 15. Data dictionary

Schemas of every artifact. All dates are timezone-aware `America/New_York` unless marked *(naive date)*.

**`data/raw/manifest.csv`** — the provenance ledger (committed to git; the files themselves are not)

| Column | Type | Meaning |
|---|---|---|
| `url` | str | exact source URL fetched |
| `raw_path` | str | repo-relative destination |
| `http_status` | int | response code (200 expected) |
| `n_bytes` | int | response body size |
| `sha256` | str | hex digest **of the raw bytes**, pre-decoding |
| `fetched_at_utc` | timestamp | when we hit the server |

**`data/interim/documents.parquet`** — one row per document

| Column | Type | Meaning |
|---|---|---|
| `event_date` | date *(naive)* | decision day (day 2 of the meeting). **Display/join only — never trades.** |
| `doc_type` | str | `"statement"` \| `"minutes"` |
| `url`, `raw_path` | str | provenance link back to the manifest |
| `text` | str | parsed body |
| `n_chars` | int | length; the flag for suspiciously short parses |
| `release_datetime` | timestamp (tz) | **the only timestamp permitted to drive trading** |
| `release_time_source` | str | `"scheduled_1400ET"` \| `"manual_override"` |

**`data/processed/spy_prices.parquet`** — indexed by trading date *(naive)*; columns `Open/High/Low/Close/Volume`, all split- and dividend-adjusted (`auto_adjust=True`). **This index is the trading-day calendar.**

**`data/processed/sentence_scores.parquet`** — one row per sentence; `doc_id`, `sentence_idx`, `sentence`, `p_pos`, `p_neg`, `p_neu`, `label`, `n_tokens`. Feeds the §8.3 extreme-sentence audit without re-running FinBERT.

**`data/processed/sentiment_scores.csv`** — one row per document; `doc_id`, `event_date`, `S_count`, `S_prob`, `n_sentences`, `n_pos`, `n_neg`, `n_neu`, `mean_confidence`. **The expensive cache.**

**`data/processed/panel.parquet`** — the analysis table, one row per event. 🔧 As actually built (n = 225):

| Column | Added in | Meaning |
|---|---|---|
| `doc_date` | 1 | date in the URL. Empirically the **publication** date, not the meeting date (a conference call on the 7th can publish on the 8th). Replaces the plan's `event_date`. |
| `meeting_heading` | 1 | the index heading verbatim, e.g. `"October 7 Conference Call - 2008"` |
| `is_scheduled` | 1 | derived from the heading; `False` for conference calls / unscheduled / notation votes |
| `release_datetime` | 1 | **tz-aware `America/New_York`** — the only timestamp allowed to drive trading |
| `release_time_source` | 1 | `manual_override` \| `parsed_from_document` \| `scheduled_default` |
| `parsed_release_time` | 1 | what the document said about itself, kept even when an override wins, so disagreements are auditable |
| `entry_date` | 1 | first session whose **09:30 open** is strictly after `release_datetime` |
| `url`, `raw_path`, `source_index`, `anchor_text` | 1 | provenance back to the manifest and the index page |
| `n_chars`, `selector_used`, `is_error_page`, `is_flagged` | 1 | parse quality; `is_flagged` = too short **or** a soft-404 body |
| `fwd_ret_1/3/5/10/20` | 1 | open-to-open adjusted returns, *h* in **sessions** |
| `S_count`, `S_prob` | 3 | raw document sentiment |
| `mu`, `sigma`, `Z` | 3 | trailing stats and the surprise; NaN for the first *L* |
| `position` | 3 | −1 / 0 / +1 from the threshold rule |

**`data/processed/spy_prices_meta.json`** — provenance sidecar for the price download: ticker, `auto_adjust`, yfinance version, UTC download time, row count, first/last session. Necessary because `period="max"` makes the result depend on the date it was fetched.

**`data/interim/discovery.parquet`** — the crawl plan, cached so the parser can be improved and re-run with **no network access at all**.

---

## 16. Revision log

Every entry records what changed and why. Per §0.2, no substantive edit to this document lands without a line here.

| Date | Commit | Change |
|---|---|---|
| 2026-07-23 | — | Initial plan authored (§§0–12 + Appendix A). |
| 2026-08-04 | *(see `git log`)* | **Phase 0 executed.** Repo scaffolded, environment built and verified, config written and validated, 31 tests passing, GPU confirmed by real kernel launch. |
| 2026-08-04 | *(see `git log`)* | **Phase 1 executed.** SPY prices (8,434 sessions, NYSE cross-check clean); 393 documents scraped with sha256 provenance; `panel.parquet` built, **n = 225**; 51 tests passing. Corrected §3.1 (disclosure regime, sample start 1994→2000), §3.3.1 (session-open entry rule + release times parsed from documents), §3.5 (measured funnel, n=219 usable), §4.3 (**overlap was understated** — 16 pairs at *h*=20, min gap 3 sessions), §7 Phase 1 → complete, §15 (real schema). Config: `start_date` 2000-01-01, `scrape_start_date`, `sample.include_unscheduled`, `min_text_chars`, `session_open_time_et`. |
| 2026-08-04 | *(this session)* | **Plan reconciled with reality.** Corrected §3.4 (`Adj Close` cannot yield an adjusted Open → `auto_adjust: true`) and §3.3 (`release_date + 1` is wrong → tz-aware timestamps + `searchsorted`). Updated §3.5 (n ≈ 263, not 240, and added the minimum-detectable-effect framing), §6.1 (actual layout, four deviations), §6.2 (Python 3.12, uv + lockfile, cu128/Blackwell, transformers 5.x), §6.3 (typed config), §7 (Phase 0 marked complete with verification commands; Phase 1 reordered and detailed with the three URL eras), §10 (risk register rebuilt with status column and six new risks), §11 (glossary tripled, grouped), Appendix A (matches `config.yaml`). Added §2.3 FinBERT label-ordering trap and the observed hawkish→positive misread. **New:** §0.1–0.2 (document roles and maintenance protocol), §13 (component reference), §14 (methodology deep-dive incl. power analysis and the Fundamental Law), §15 (data dictionary), §16 (this log). **New file:** `STATUS.md`. |

---

*End of plan. Phase 0 is green; Phase 1 is next. Do not advance a phase until its Definition-of-Done is green — that discipline is the project's defense against the cardinal sin. And per §0.2: when the code changes, this document changes in the same session.*
