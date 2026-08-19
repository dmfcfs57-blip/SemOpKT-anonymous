"""Machine-readable run provenance without collecting personal identity."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _command_output(command: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def git_revision(repository: str | Path) -> dict[str, Any]:
    root = Path(repository).resolve()
    return {
        "commit": _command_output(["git", "rev-parse", "HEAD"], root),
        "dirty": bool(_command_output(["git", "status", "--porcelain"], root)),
    }


def environment_record(repository: str | Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor_architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "git": git_revision(repository),
    }
    try:
        import numpy
        import pandas
        import sklearn
        import torch

        record["libraries"] = {
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        }
        record["accelerator"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
    except ImportError:
        record["libraries"] = {}
        record["accelerator"] = {"cuda_available": False, "device": "unavailable"}
    return record

