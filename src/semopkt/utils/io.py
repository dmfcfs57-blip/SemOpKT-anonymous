"""Atomic serialization helpers and table format negotiation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def atomic_write_text(path: str | Path, text: str) -> None:
    target = ensure_parent(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_yaml(path: str | Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(path, yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True))


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(source)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(source, sep="\t")
    raise ValueError(f"Unsupported table format: {source}")


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    target = ensure_parent(path)
    if target.suffix.lower() == ".parquet":
        frame.to_parquet(target, index=False)
    elif target.suffix.lower() in {".tsv", ".txt"}:
        frame.to_csv(target, sep="\t", index=False, lineterminator="\n")
    else:
        frame.to_csv(target, index=False, lineterminator="\n")

