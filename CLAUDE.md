# Working agreements — fed-sentiment-alpha

Read [PLAN.md](PLAN.md) for the complete picture and [STATUS.md](STATUS.md) for the current position.

## Git: never run it (highest priority rule)

**Aryan runs every git command himself.** Never run `init`, `add`, `commit`, `branch`, `checkout`, `stash`, `push`, `pull`, `merge`, `rebase`, `tag` or `reset`, and never open a PR. Make file edits only and leave the working tree modified.

When a change is ready, **end the response with a fenced `bash` block containing the exact `git add` and `git commit` commands** for him to run.

**Never add a `Co-Authored-By` trailer** to a commit message. Detailed, well-reasoned commit messages are still wanted — they are simply his to run.

Read-only git (`git status`, `git log`, `git diff`, `git show`) is fine for inspection.

## The documentation rule (non-negotiable)

**Every change to code, environment, data, or findings must be reflected in the documentation in the same session that makes the change.** This is [PLAN.md §0.2](PLAN.md). Specifically, before a session ends:

1. **Correct affected lines in PLAN.md in place** — do not append a "but actually…" note elsewhere. If §3.4 describes behaviour that changed, §3.4 changes.
2. **Mark deviations from the original plan with 🔧 and state the reason.** A deviation without a reason reads as drift; with a reason it reads as research. The reasons are interview material.
3. **Rewrite the affected parts of STATUS.md** — phase board, what is done, findings, next objectives.
4. **Append a row to PLAN.md §16 (revision log)** with date, commit, and a one-line summary.
5. **Turn the phase's DoD in §7 from a promise into a record** of what was verified, including the command that verified it.

Rationale: in a research project the document *is* the deliverable and the code is evidence for it. Every hour PLAN.md and the repo disagree is an hour spent building on a false model of the work — the same category of error as look-ahead bias.

## Project conventions

- **Phase discipline.** Do not start phase *N+1* until phase *N*'s Definition of Done in PLAN.md §7 is green.
- **`config.yaml` is pre-registration.** Changing a primary value after seeing results must fail `tests/test_config.py::test_preregistered_primary_values`. If you need to change one, say so explicitly and record why — never edit quietly.
- **Never tune toward a positive result.** A rigorously demonstrated null is the intended, acceptable outcome (PLAN.md §5.5). Massaging the pipeline until an edge appears *is* the overfitting this project exists to avoid.
- **`data/raw` is immutable.** Written once, hashed into `manifest.csv`, never edited.
- **Failures are loud; degradation is flagged.** Nothing is silently dropped, defaulted, or coerced. A document that will not parse is counted and reported — never scored as neutral.
- **Every transformation states what timestamp its inputs are valid as-of.**

## Explanation style

Aryan is learning quant finance and ML alongside building this. Write for that:

- **Define every term at first use in each file** — in docstrings and config comments, not only in chat. Some repetition across files is correct.
- **Put the *why* next to the code.** Module docstrings carry the concept teaching; config comments carry the rationale for each value; commit messages explain deviations.
- **Prefer a concrete counterexample to an asserted rule.** "`release_date + 1` fails because 2020-03-15 was a Sunday" beats "use a trading calendar."

## Environment

```bash
uv sync                                # Python 3.12 + locked deps
uv run pytest                          # 31 passed, 15 skipped (canaries for Phases 1-3)
uv run python scripts/check_gpu.py     # must PASS the real kernel launch
uv run ruff check . && uv run ruff format .
```

- **torch must come from the `cu128` index.** The RTX 5070 Ti is `sm_120` (Blackwell); earlier CUDA builds pass every preliminary check and then fail at the first kernel launch. Never "fix" a torch problem by installing from PyPI's default index.
- **Never create `src/signal.py`** — it shadows the stdlib `signal` module that torch imports at startup. The file is `src/alpha_signal.py`.
- **Index FinBERT outputs by name**, from `model.config.id2label` (`{0: positive, 1: negative, 2: neutral}`), never by position. Positional indexing silently inverts the signal.
