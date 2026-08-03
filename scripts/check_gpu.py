"""Report the CUDA situation and prove a kernel actually launches.

Run with:  uv run python scripts/check_gpu.py

Why this exists
---------------
The RTX 50-series (Blackwell) has compute capability **sm_120**. A PyTorch wheel
only runs on a GPU whose ``sm_XX`` appears in the wheel's compiled arch list, or
whose architecture it can reach by JIT-compiling embedded PTX. Wheels from the
``cu126`` index and earlier contain no sm_120 kernels, so a fresh
``pip install torch`` from PyPI -- which serves the *default* CUDA variant --
can install cleanly, report ``torch.cuda.is_available() == True``, and then fail
at the first matmul with::

    RuntimeError: CUDA error: no kernel image is available for execution on the device

That failure mode is confusing precisely because every preliminary check passes.
This script runs the only check that settles it: an actual kernel launch.

Glossary
--------
compute capability
    NVIDIA's versioning of a GPU's instruction set, written ``sm_XX``.
    RTX 5070 Ti (GB203) is ``sm_120``.
cubin
    Pre-compiled GPU machine code for one specific ``sm_XX`` target.
PTX
    A forward-compatible intermediate assembly the driver can JIT-compile for a
    newer architecture. Slow to start and not always present.
CUDA runtime vs driver
    The *driver* ships with your NVIDIA display driver; the *runtime* is bundled
    inside the PyTorch wheel. No system CUDA Toolkit install is required. A
    newer driver runs older runtimes (your 13.3 driver runs a 12.8 runtime); the
    reverse does not hold.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch is not installed. Run `uv sync`.")
        return 1

    print(f"python           : {sys.version.split()[0]}")
    print(f"torch            : {torch.__version__}")
    print(f"built with CUDA  : {torch.version.cuda or 'CPU-only build'}")
    print(f"cuda available   : {torch.cuda.is_available()}")

    if torch.version.cuda is None:
        print(
            "\nFAIL: this is a CPU-only torch build.\n"
            "      pyproject.toml pins torch to the cu128 index; re-run `uv sync`."
        )
        return 1

    if not torch.cuda.is_available():
        print("\nFAIL: CUDA build installed but no device is visible to torch.")
        print("      Check `nvidia-smi` and that the driver is current.")
        return 1

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    arch_list = torch.cuda.get_arch_list()

    print(f"device           : {name}")
    print(f"compute capability: sm_{major}{minor}")
    print(f"total memory     : {total_gb:.1f} GiB")
    print(f"wheel arch list  : {', '.join(arch_list)}")

    target = f"sm_{major}{minor}"
    if target in arch_list:
        print(f"\n  native cubins present for {target}")
    else:
        print(
            f"\n  WARNING: {target} is NOT in this wheel's arch list.\n"
            f"  If it runs at all it will be via slow PTX JIT, and it may not run.\n"
            f"  Install from https://download.pytorch.org/whl/cu128 or newer."
        )

    # The only check that actually proves anything.
    print("\nlaunching a real kernel (512x512 matmul)...")
    try:
        a = torch.randn(512, 512, device="cuda")
        out = (a @ torch.eye(512, device="cuda")).cpu()
        torch.testing.assert_close(out, a.cpu(), rtol=1e-4, atol=1e-4)
    except Exception as exc:
        print(f"FAIL: kernel launch failed: {exc}")
        return 1

    print("PASS: kernel executed and returned a numerically correct result.")
    print("\nGPU is ready for FinBERT inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
