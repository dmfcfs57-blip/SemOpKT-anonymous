"""Configuration loading, validation, hashing, and override support."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"Configuration must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_overrides(config: Mapping[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    for expression in overrides:
        if "=" not in expression:
            raise ConfigError(f"Override must be key=value: {expression}")
        dotted_key, raw_value = expression.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor: dict[str, Any] = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigError(f"Cannot set nested override below {part}")
            cursor = child
        cursor[parts[-1]] = value
    return result


def canonical_config(config: Mapping[str, Any]) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: clean(child)
                for key, child in value.items()
                if not str(key).startswith("_")
            }
        if isinstance(value, list):
            return [clean(child) for child in value]
        if isinstance(value, tuple):
            return tuple(clean(child) for child in value)
        return copy.deepcopy(value)

    return clean(config)


def config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        canonical_config(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require(config: Mapping[str, Any], *paths: str) -> None:
    missing: list[str] = []
    for dotted in paths:
        cursor: Any = config
        for part in dotted.split("."):
            if not isinstance(cursor, Mapping) or part not in cursor:
                missing.append(dotted)
                break
            cursor = cursor[part]
    if missing:
        raise ConfigError("Missing configuration keys: " + ", ".join(sorted(set(missing))))
