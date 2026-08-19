"""Validation-only scalar temperature scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class TemperatureScaler:
    temperature: float
    validation_count: int

    @classmethod
    def fit(cls, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        if len(logits) == 0 or len(logits) != len(labels):
            raise ValueError("Temperature calibration requires aligned non-empty logits and labels")

        def objective(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            scaled = logits / temperature
            return float(np.mean(np.logaddexp(0.0, scaled) - labels * scaled))

        result = minimize_scalar(
            objective,
            method="bounded",
            bounds=(np.log(0.05), np.log(20.0)),
            options={"xatol": 1.0e-8},
        )
        if not result.success:
            raise RuntimeError(f"Temperature optimization failed: {result.message}")
        return cls(float(np.exp(result.x)), int(len(logits)))

    def apply_logits(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=np.float64) / self.temperature

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        scaled = self.apply_logits(logits)
        return 1.0 / (1.0 + np.exp(-np.clip(scaled, -40.0, 40.0)))

