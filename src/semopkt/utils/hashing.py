"""Stable SHA-256 hashing for files, tables, configurations, and source trees."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_json(value: Any) -> str:
    return hash_bytes(canonical_json(value).encode("utf-8"))


def hash_dataframe(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    selected = frame.loc[:, list(columns)] if columns is not None else frame
    normalized = selected.copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].map(
            lambda value: canonical_json(value)
            if isinstance(value, (dict, list, tuple, set))
            else "" if pd.isna(value) else str(value)
        )
    normalized = normalized.sort_values(list(normalized.columns), kind="mergesort")
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hash_bytes(payload)


def hash_tree(
    root: str | Path,
    suffixes: tuple[str, ...] = (".py", ".yaml", ".yml", ".toml"),
    excluded_parts: tuple[str, ...] = (".git", "runs", "generated", "__pycache__"),
) -> str:
    root_path = Path(root).resolve()
    records: list[tuple[str, str]] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        relative = path.relative_to(root_path)
        if any(part in excluded_parts for part in relative.parts):
            continue
        records.append((relative.as_posix(), hash_file(path)))
    return hash_json(records)


def bind_hashes(values: Mapping[str, str]) -> str:
    return hash_json(dict(sorted(values.items())))

