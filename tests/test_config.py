"""Phase 0: config.yaml parses, validates, and rejects bad input.

``config.yaml`` is the auditable record of pre-registration (PLAN.md 5.1). A
config that can be *silently* wrong -- a typo'd key falling back to a default,
say -- would undermine the central claim that the reported result corresponds to
the pre-declared parameters. These tests assert the validation actually bites.
"""

from __future__ import annotations

import copy
from datetime import date, time

import pytest
import yaml
from pydantic import ValidationError

from src.config import REPO_ROOT, Config, load_config


@pytest.fixture(scope="module")
def raw_yaml() -> dict:
    with (REPO_ROOT / "config.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_config_loads() -> None:
    cfg = load_config()
    assert isinstance(cfg, Config)


def test_preregistered_primary_values() -> None:
    """Pin the pre-registered primary configuration (PLAN.md 5.1, Appendix A).

    If someone edits a primary value after seeing results, this test fails and
    forces the change to be a conscious, reviewed act rather than a quiet one.
    """
    cfg = load_config()
    assert cfg.sentiment.aggregation == "prob"
    assert cfg.signal.zscore_window_L == 6
    assert cfg.signal.threshold_theta == 1.0
    assert cfg.signal.hypothesis_sign == "dovish_surprise_bullish"
    assert cfg.backtest.primary_horizon_days == 5
    assert cfg.backtest.entry == "next_trading_open"
    assert cfg.evaluation.ic_method_primary == "spearman"


def test_price_fields_are_consistent() -> None:
    """auto_adjust must be on whenever an Open is used as a trade price.

    With ``auto_adjust=False`` yfinance provides no adjusted Open, so entering
    on ``Open`` would mix an unadjusted entry with an adjusted exit and inject
    every SPY dividend (~1.3%/yr) into the return as a phantom loss.
    """
    cfg = load_config()
    if "Open" in (cfg.data.entry_price_field, cfg.data.exit_price_field):
        assert cfg.data.auto_adjust is True


def test_release_overrides_parsed_as_datetimes(raw_yaml: dict) -> None:
    """The unscheduled-meeting overrides must survive YAML round-tripping.

    2020-03-15 (Sunday, 17:00 ET) is the case that breaks naive
    ``release_date + 1 day`` arithmetic, so its presence is worth asserting.
    """
    cfg = load_config()
    overrides = cfg.scrape.release_time_overrides
    assert date(2020, 3, 15) in overrides
    assert date(2008, 10, 8) in overrides
    assert all(isinstance(v, time) for v in overrides.values())
    # The pre-open 2008 release must not be confused with the 14:00 default.
    assert overrides[date(2008, 10, 8)] < cfg.scrape.scheduled_release_time_et


def test_unknown_key_is_rejected(raw_yaml: dict) -> None:
    """extra='forbid' turns a typo into an error instead of a silent default."""
    bad = copy.deepcopy(raw_yaml)
    bad["signal"]["zscore_window_l"] = 12  # lowercase L -- a plausible typo
    with pytest.raises(ValidationError):
        Config.model_validate(bad)


def test_primary_horizon_must_be_in_grid(raw_yaml: dict) -> None:
    """A headline horizon absent from the reported grid would hide the result."""
    bad = copy.deepcopy(raw_yaml)
    bad["backtest"]["primary_horizon_days"] = 7  # not in [1,3,5,10,20]
    with pytest.raises(ValidationError):
        Config.model_validate(bad)


def test_horizons_must_be_ascending(raw_yaml: dict) -> None:
    bad = copy.deepcopy(raw_yaml)
    bad["backtest"]["horizons_days"] = [5, 1, 3, 10, 20]
    with pytest.raises(ValidationError):
        Config.model_validate(bad)


def test_zscore_window_needs_at_least_two_points(raw_yaml: dict) -> None:
    """A standard deviation over one observation is undefined."""
    bad = copy.deepcopy(raw_yaml)
    bad["signal"]["zscore_window_L"] = 1
    with pytest.raises(ValidationError):
        Config.model_validate(bad)


def test_max_length_capped_at_finbert_limit(raw_yaml: dict) -> None:
    """BERT's positional embedding table has exactly 512 rows."""
    bad = copy.deepcopy(raw_yaml)
    bad["sentiment"]["max_length"] = 1024
    with pytest.raises(ValidationError):
        Config.model_validate(bad)


def test_paths_resolve_under_repo_root() -> None:
    cfg = load_config()
    for key in ("data_raw", "data_interim", "data_processed", "reports"):
        resolved = cfg.paths.resolve(key)
        assert resolved.is_absolute()
        assert REPO_ROOT in resolved.parents or resolved.parent == REPO_ROOT
