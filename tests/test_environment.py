"""Phase 0 Definition-of-Done: the environment is actually usable.

PLAN.md section 7, Phase 0 DoD: ``python -c "import torch, transformers,
yfinance, bs4"`` succeeds. This asserts that, plus the CUDA facts specific to
this machine, so a broken install fails here rather than 40 minutes into a
FinBERT pass.
"""

from __future__ import annotations

import importlib

import pytest

# (import name, human-readable purpose)
REQUIRED = [
    ("pandas", "dataframes"),
    ("numpy", "numerics"),
    ("pyarrow", "parquet i/o"),
    ("yaml", "config loading"),
    ("pydantic", "config validation"),
    ("yfinance", "SPY price download"),
    ("pandas_market_calendars", "NYSE calendar validation"),
    ("requests", "http"),
    ("bs4", "html parsing"),
    ("lxml", "html parser backend"),
    ("torch", "FinBERT backend"),
    ("transformers", "FinBERT model loading"),
    ("nltk", "sentence tokenisation"),
    ("scipy", "spearmanr, bootstrap"),
    ("sklearn", "confusion matrix"),
    ("statsmodels", "robust standard errors"),
    ("matplotlib", "plots"),
    ("seaborn", "plots"),
]


@pytest.mark.parametrize("module_name,purpose", REQUIRED)
def test_dependency_importable(module_name: str, purpose: str) -> None:
    try:
        importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - only fires on a broken env
        pytest.fail(f"{module_name!r} ({purpose}) is not importable: {exc}")


def test_python_version() -> None:
    """3.12 is pinned: torch, quantstats and statsmodels all have solid wheels."""
    import sys

    assert sys.version_info[:2] == (3, 12), f"expected Python 3.12, got {sys.version}"


def test_torch_cuda_build() -> None:
    """The installed torch must be a CUDA build compiled for Blackwell.

    An RTX 50-series GPU has compute capability ``sm_120``. Only PyTorch wheels
    from the ``cu128`` index or newer contain sm_120 kernels; a cu126-or-earlier
    build raises "no kernel image is available for execution on the device" at
    the first forward pass.

    Skipped rather than failed when no GPU is present, so the suite stays green
    on a CPU-only reviewer machine.
    """
    import torch

    if torch.version.cuda is None:
        pytest.skip("CPU-only torch build installed; GPU path not exercised")
    if not torch.cuda.is_available():
        pytest.skip("CUDA build present but no device visible")

    major, minor = torch.cuda.get_device_capability(0)
    capability = major * 10 + minor
    arch_list = torch.cuda.get_arch_list()

    if capability >= 120:
        assert any("sm_120" in a or "sm_121" in a for a in arch_list), (
            f"GPU is sm_{capability} (Blackwell) but this torch build only targets "
            f"{arch_list}. Reinstall from the cu128 index or newer."
        )


@pytest.mark.gpu
def test_cuda_matmul_actually_runs() -> None:
    """End-to-end proof: a real kernel launches and returns a correct result.

    ``get_arch_list()`` only reports what the wheel *claims*. This executes a
    matmul on the device, which is the only way to be certain the kernels load.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    a = torch.randn(512, 512, device="cuda")
    result = (a @ torch.eye(512, device="cuda")).cpu()
    torch.testing.assert_close(result, a.cpu(), rtol=1e-4, atol=1e-4)
