"""Serializable run specifications."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from semopkt.utils.hashing import hash_json


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    dataset: str
    protocol: str
    model: str
    seed: int
    train_size: int | None = None
    holdout_ratio: float | None = None
    source_dataset: str | None = None
    target_dataset: str | None = None
    adaptation_size: int | None = None
    condition: str = "default"
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> str:
        pieces = [
            self.experiment,
            self.dataset,
            self.protocol,
            self.model,
            str(self.seed),
            f"k-{self.train_size if self.train_size is not None else 'full'}",
        ]
        if self.holdout_ratio is not None:
            pieces.append(f"r-{int(round(self.holdout_ratio * 100))}")
        if self.source_dataset:
            pieces.append(f"source-{self.source_dataset}")
        if self.adaptation_size is not None:
            pieces.append(f"adapt-{self.adaptation_size}")
        if self.condition != "default":
            pieces.append(self.condition.replace("/", "-"))
        pieces.append(hash_json(self.to_dict())[:10])
        return "__".join(pieces)

