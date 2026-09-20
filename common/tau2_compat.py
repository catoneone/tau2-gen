"""Locate the tau2-bench clone and its virtualenv, and build a telecom environment over a custom DB.

The generator only depends on tau2-bench's telecom environment, data models, task dataclasses and user
simulator prompt.

tau2-bench requires Python 3.12/3.13, so run everything through the clone's own interpreter:
    upstream/tau2-bench/.venv/bin/python domains/telecom/gen.py ...
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # the tau2-gen checkout
GEN_ROOT = REPO_ROOT  # kept as a separate name: generated output is rooted here
TAU2_ROOT = Path(os.environ.get("TAU2_ROOT", REPO_ROOT / "upstream" / "tau2-bench"))
TAU2_PYTHON = TAU2_ROOT / ".venv" / "bin" / "python"
TAU2_TELECOM_DATA = TAU2_ROOT / "data" / "tau2" / "domains" / "telecom"


def bootstrap() -> None:
    """Make `import common` work when a script is run directly by path."""
    if str(GEN_ROOT) not in sys.path:
        sys.path.insert(0, str(GEN_ROOT))


def require_tau2() -> None:
    try:
        import tau2  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            f"cannot import tau2 ({e}). Run this with {TAU2_PYTHON}; see README.md for setup."
        ) from e


def tau2_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(TAU2_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def repo_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def build_env(db: dict | None = None):
    """A tau2 telecom environment. db=None uses tau2's own db.toml; otherwise the given DB dict."""
    from tau2.domains.telecom.data_model import TelecomDB
    from tau2.domains.telecom.environment import get_environment

    return get_environment(db=TelecomDB.model_validate(db) if db is not None else None)
