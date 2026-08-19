"""Canonical interaction schema and text normalization."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Iterable, Mapping, Sequence

import pandas as pd

from semopkt.constants import STANDARD_COLUMNS

_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return _WHITESPACE.sub(" ", text).strip()


def split_components(value: object, delimiters: Sequence[str]) -> list[str]:
    raw = str(value)
    pattern = "|".join(re.escape(delimiter) for delimiter in delimiters if delimiter)
    parts = re.split(pattern, raw) if pattern else [raw]
    normalized: list[str] = []
    seen: set[str] = set()
    for part in parts:
        item = normalize_text(part)
        if item and item not in seen:
            normalized.append(item)
            seen.add(item)
    return normalized


def canonical_descriptor(components: Iterable[str], joiner: str = " [SEP] ") -> str:
    return joiner.join(component.strip() for component in components if component.strip())


def stable_identifier(text: str, prefix: str = "kc") -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def resolve_column(columns: Iterable[str], aliases: Sequence[str], field: str) -> str:
    available = list(columns)
    direct = {column: column for column in available}
    folded = {normalize_text(column): column for column in available}
    for alias in aliases:
        if alias in direct:
            return direct[alias]
        if normalize_text(alias) in folded:
            return folded[normalize_text(alias)]
    raise KeyError(f"No column for {field}; tried {aliases}; available columns are {available}")


def parse_component_cell(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [part for part in text.split(" [SEP] ") if part]


def validate_standard_frame(frame: pd.DataFrame) -> None:
    missing = sorted(set(STANDARD_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing standard columns: {missing}")
    if frame.empty:
        raise ValueError("Interaction table is empty")
    labels = set(frame["correct"].dropna().astype(int).unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Correctness must be binary, observed {sorted(labels)}")
    if frame[["student_id", "question_id", "kc_id"]].isna().any().any():
        raise ValueError("Identifiers may not be missing")
    positions = frame.groupby("student_id", sort=False)["position"].apply(list)
    for student_id, values in positions.items():
        expected = list(range(1, len(values) + 1))
        if list(map(int, values)) != expected:
            raise ValueError(f"Non-contiguous positions for student {student_id}")


def interaction_contains_heldout(component_cell: object, heldout: set[str]) -> bool:
    return bool(set(parse_component_cell(component_cell)) & heldout)

