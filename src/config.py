"""Typed, validated loader for ``config.yaml``.

Why a schema instead of ``yaml.safe_load()`` returning a plain dict?

A dict fails *silently*. Misspell ``zscore_window_L`` as ``zscore_window_l`` and
``cfg["signal"]["zscore_window_L"]`` raises a KeyError three phases later --
or worse, a ``.get(..., 6)`` default swallows it and you unknowingly report a
result from the wrong configuration. Since PLAN.md section 5.1 makes
``config.yaml`` the *auditable record of pre-registration*, a config that can be
silently wrong undermines the central claim of the project.

Pydantic validates at load time: unknown keys, wrong types, and out-of-range
values all raise immediately, with the offending field named.

Terms used below
----------------
pydantic
    A Python library that builds runtime-validated classes from type hints.
    ``BaseModel`` subclasses parse-and-validate their inputs on construction.
extra="forbid"
    Pydantic setting: reject any key not declared on the model. This is what
    turns a typo from a silent no-op into a loud error.
frozen=True
    Makes instances immutable. Config should never be mutated mid-run -- that
    would mean the results no longer correspond to the file on disk.
"""

from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Repository root = the directory containing this file's parent (src/ -> repo/).
# Resolved from __file__ rather than os.getcwd() so the code works identically
# whether launched from the repo root, from notebooks/, or by pytest.
REPO_ROOT = Path(__file__).resolve().parent.parent


class _Base(BaseModel):
    """Shared strictness settings for every config section."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Base):
    start_date: date
    end_date: date | None = None  # None => "today", resolved at download time
    ticker: str
    auto_adjust: bool
    entry_price_field: Literal["Open", "Close"]
    exit_price_field: Literal["Open", "Close"]


class ScrapeConfig(_Base):
    base_url: str
    calendar_url: str
    historical_url_template: str
    historical_year_start: int = Field(ge=1990)
    historical_year_end: int = Field(ge=1990)
    user_agent: str
    request_delay_seconds: float = Field(ge=0.0)
    timeout_seconds: int = Field(gt=0)
    max_retries: int = Field(ge=0)
    document_types: list[Literal["statement", "minutes"]]
    scheduled_release_time_et: time
    # Keys are ISO dates as written in YAML; PyYAML parses bare 2008-10-08 into
    # a date object, so the annotation is dict[date, time] rather than str keys.
    release_time_overrides: dict[date, time] = Field(default_factory=dict)


class SentimentConfig(_Base):
    model: str
    aggregation: Literal["prob", "count"]
    max_length: int = Field(gt=0, le=512)  # FinBERT's positional embeddings stop at 512
    batch_size: int = Field(gt=0)
    device: Literal["auto", "cuda", "cpu"]
    seed: int
    strip_boilerplate: bool


class SignalConfig(_Base):
    zscore_window_L: int = Field(ge=2)  # need >= 2 points for a standard deviation
    threshold_theta: float = Field(ge=0.0)
    hypothesis_sign: Literal["dovish_surprise_bullish", "dovish_surprise_bearish"]


class BacktestConfig(_Base):
    primary_horizon_days: int = Field(gt=0)
    horizons_days: list[int]
    entry: Literal["next_trading_open", "same_day_close"]

    @field_validator("horizons_days")
    @classmethod
    def _positive_and_sorted(cls, v: list[int]) -> list[int]:
        if any(h <= 0 for h in v):
            raise ValueError("all horizons must be positive trading-day counts")
        if v != sorted(v):
            raise ValueError("horizons_days must be ascending (IC-decay plots assume order)")
        return v


class CostsConfig(_Base):
    bps_grid: list[float]
    cost_model: Literal["per_unit_turnover"]


class EvaluationConfig(_Base):
    ic_method_primary: Literal["spearman", "pearson"]
    bootstrap_iters: int = Field(gt=0)
    bootstrap_ci: float = Field(gt=0.0, lt=1.0)
    random_seed: int


class RobustnessConfig(_Base):
    zscore_window_L: list[int]
    threshold_theta: list[float]
    aggregation: list[Literal["prob", "count"]]
    entry: list[Literal["next_trading_open", "same_day_close"]]


class PathsConfig(_Base):
    data_raw: str
    data_interim: str
    data_processed: str
    reports: str

    def resolve(self, key: str) -> Path:
        """Return an absolute Path for one of the configured directories.

        Paths in config.yaml are relative to the repo root so the file stays
        machine-independent; this turns them absolute at use time.
        """
        return REPO_ROOT / getattr(self, key)


class Config(_Base):
    data: DataConfig
    scrape: ScrapeConfig
    sentiment: SentimentConfig
    signal: SignalConfig
    backtest: BacktestConfig
    costs: CostsConfig
    evaluation: EvaluationConfig
    robustness: RobustnessConfig
    paths: PathsConfig

    @field_validator("backtest")
    @classmethod
    def _primary_horizon_in_grid(cls, v: BacktestConfig) -> BacktestConfig:
        if v.primary_horizon_days not in v.horizons_days:
            raise ValueError(
                f"primary_horizon_days={v.primary_horizon_days} must appear in "
                f"horizons_days={v.horizons_days}; otherwise the headline result "
                "would not be visible in the reported grid."
            )
        return v


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate ``config.yaml``.

    Parameters
    ----------
    path
        Optional explicit path. Defaults to ``<repo root>/config.yaml``.

    Raises
    ------
    pydantic.ValidationError
        If any field is missing, mistyped, unknown, or out of range.
    """
    cfg_path = Path(path) if path is not None else REPO_ROOT / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
