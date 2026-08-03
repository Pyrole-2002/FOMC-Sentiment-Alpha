"""Pytest root configuration.

Its mere presence at the repository root makes ``import src...`` work: under
pytest's default ``importmode=prepend``, the directory containing the rootdir
conftest is prepended to ``sys.path``. The explicit insertion below makes that
guarantee independent of pytest version and invocation directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
