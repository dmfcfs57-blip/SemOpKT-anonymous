"""Deterministic early-stopped training with interaction-level predictions."""

from __future__ import annotations

import copy
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from semopkt.data.sequences import StudentSequence, TorchSequenceDataset, collate_sequences
from semopkt.evaluation.metrics import compute_metrics
from semopkt.models.base import KTModel
from semopkt.utils.hashing import hash_file
from semopkt.utils.io import write_json


@dataclass
class TrainingResult:
    best_epoch: int
    history: list[dict[str, float]]
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    test_metrics: dict[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str
    train_seconds: float
    peak_memory_bytes: int


def _device_from_name(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self,
        model: KTModel,
        training_config: Mapping[str, Any],
        run_directory: str | Path,
        device: str | None = None,
    ):
        self.model = model
        self.config = dict(training_config)
        self.run_directory = Path(run_directory)
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.device = _device_from_name(device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.get("learning_rate", 1.0e-3)),
            weight_decay=float(self.config.get("weight_decay", 1.0e-4)),
        )
        self.gradient_clip = float(self.config.get("gradient_clip", 5.0))
        self.batch_size = int(self.config.get("batch_size", 64))
        self.num_workers = int(self.config.get("num_workers", 0))
        self.precision = str(self.config.get("precision", "fp32")).casefold()

    def _autocast(self):
        if self.device.type == "cuda" and self.precision in {"bf16", "bfloat16"}:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _loader(self, sequences: Sequence[StudentSequence], shuffle: bool) -> DataLoader:
        return DataLoader(
            TorchSequenceDataset(sequences),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_sequences,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

    def _move(self, batch: Mapping[str, object]) -> dict[str, object]:
        return {
            key: value.to(self.device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def prepare_batch(self, batch: Mapping[str, object]) -> dict[str, object]:
        """Move a collated batch to the configured execution device."""

        return self._move(batch)

    def _epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        totals = {"prediction": 0.0, "smoothness": 0.0, "stability": 0.0, "total": 0.0}
        batches = 0
        for raw_batch in loader:
            batch = self._move(raw_batch)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                output = self.model(batch)  # type: ignore[arg-type]
                loss, parts = self.model.loss(output, batch["labels"])  # type: ignore[arg-type]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            self.model.normalize_parameters()
            for key in totals:
                totals[key] += parts[key]
            batches += 1
        return {key: value / max(1, batches) for key, value in totals.items()}

    @torch.inference_mode()
    def predict(self, sequences: Sequence[StudentSequence], score_only: bool = True) -> pd.DataFrame:
        self.model.eval()
        records: list[dict[str, Any]] = []
        for raw_batch in self._loader(sequences, shuffle=False):
            batch = self._move(raw_batch)
            with self._autocast():
                output = self.model(batch)  # type: ignore[arg-type]
            logits = output.logits.detach().float().cpu().numpy()
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
            labels = batch["labels"].detach().cpu().numpy()  # type: ignore[union-attr]
            positions = batch["positions"].detach().cpu().numpy()  # type: ignore[union-attr]
            valid = batch["valid_mask"].detach().cpu().numpy().astype(bool)  # type: ignore[union-attr]
            score = batch["score_mask"].detach().cpu().numpy().astype(bool)  # type: ignore[union-attr]
            selected = valid & score if score_only else valid
            for row, student_id in enumerate(batch["student_id"]):  # type: ignore[index]
                source_ids = batch["source_row_ids"][row]  # type: ignore[index]
                question_ids = batch["question_ids"][row]  # type: ignore[index]
                kc_ids = batch["kc_ids"][row]  # type: ignore[index]
                for step in np.flatnonzero(selected[row]):
                    records.append(
                        {
                            "dataset": batch["dataset"][row],  # type: ignore[index]
                            "student_id": str(student_id),
                            "question_id": str(question_ids[step]),
                            "kc_id": str(kc_ids[step]),
                            "source_row_id": str(source_ids[step]),
                            "correct": int(labels[row, step]),
                            "logit": float(logits[row, step]),
                            "probability": float(probabilities[row, step]),
                            "position": int(positions[row, step]),
                        }
                    )
        return pd.DataFrame.from_records(records)

    def fit(
        self,
        train_sequences: Sequence[StudentSequence],
        validation_sequences: Sequence[StudentSequence],
        test_sequences: Sequence[StudentSequence],
        ece_bins: int = 15,
    ) -> TrainingResult:
        if not train_sequences or not validation_sequences or not test_sequences:
            raise ValueError("Training, validation, and test sequences must all be non-empty")
        if hasattr(self.model, "build_smoothness_graph"):
            self.model.build_smoothness_graph()  # type: ignore[attr-defined]
        maximum_epochs = int(self.config.get("max_epochs", 200))
        patience = int(self.config.get("patience", 20))
        best_loss = math.inf
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        history: list[dict[str, float]] = []
        stale = 0
        train_loader = self._loader(train_sequences, shuffle=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        for epoch in range(1, maximum_epochs + 1):
            train_metrics = self._epoch(train_loader)
            validation = self.predict(validation_sequences, score_only=False)
            validation_metrics = compute_metrics(validation, ece_bins=ece_bins)
            validation_loss = float(validation_metrics["nll"])
            history.append(
                {
                    "epoch": float(epoch),
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    "validation_nll": validation_loss,
                    "validation_auc": float(validation_metrics["auc"]),
                }
            )
            if validation_loss < best_loss - 1.0e-7:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        train_seconds = time.perf_counter() - start
        if best_state is None:
            raise RuntimeError("No valid checkpoint was selected")
        self.model.load_state_dict(best_state)
        checkpoint_path = self.run_directory / "best_model.pt"
        torch.save({"model_state": best_state, "best_epoch": best_epoch}, checkpoint_path)
        validation_predictions = self.predict(validation_sequences, score_only=False)
        test_predictions = self.predict(test_sequences, score_only=True)
        test_metrics = compute_metrics(test_predictions, ece_bins=ece_bins)
        peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else 0
        )
        write_json(self.run_directory / "history.json", history)
        write_json(self.run_directory / "metrics.json", test_metrics)
        return TrainingResult(
            best_epoch=best_epoch,
            history=history,
            validation_predictions=validation_predictions,
            test_predictions=test_predictions,
            test_metrics=test_metrics,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=hash_file(checkpoint_path),
            train_seconds=float(train_seconds),
            peak_memory_bytes=peak_memory,
        )
