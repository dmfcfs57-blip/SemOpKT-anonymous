"""Batch-one latency, throughput, memory, and parameter measurement."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
import torch

from semopkt.models.base import KTModel
from semopkt.models.semopkt import SemOpKT


@torch.inference_mode()
def measure_inference(
    model: KTModel,
    batch: Mapping[str, Any],
    warmup: int = 20,
    repetitions: int = 100,
) -> dict[str, float | int]:
    device = next(model.parameters()).device
    model.eval()
    for _ in range(warmup):
        model(batch)  # type: ignore[arg-type]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timings = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter_ns()
        model(batch)  # type: ignore[arg-type]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings[index] = (time.perf_counter_ns() - start) / 1.0e6
    return {
        "latency_median_ms": float(np.median(timings)),
        "latency_p95_ms": float(np.quantile(timings, 0.95)),
        "latency_mean_ms": float(np.mean(timings)),
        "latency_sd_ms": float(np.std(timings, ddof=1)),
        "throughput_sequences_per_second": float(1000.0 / np.mean(timings)),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "total_parameters": model.total_parameter_count(),
        "trainable_parameters": model.trainable_parameter_count(),
        "warmup": int(warmup),
        "repetitions": int(repetitions),
    }


@torch.inference_mode()
def measure_semopkt_query_scaling(
    model: SemOpKT,
    query_sizes: list[int],
    warmup: int = 20,
    repetitions: int = 100,
) -> list[dict[str, float | int]]:
    """Measure cached-coordinate field queries without first-time text encoding."""

    device = next(model.parameters()).device
    model.eval()
    coordinates = model.concept_coordinates()
    state = model.population_prior[None, :, :]
    rows: list[dict[str, float | int]] = []
    for query_count in query_sizes:
        repeats = int(np.ceil(query_count / len(coordinates)))
        query = coordinates.repeat(repeats, 1)[:query_count][None, :, :]

        def execute() -> None:
            field = model.field_query(state, query)
            inputs = torch.cat([field, query], dim=-1) if model.direct_coordinate_path else field
            model.prediction_head(inputs)

        def execute_field_only() -> None:
            model.field_query(state, query)

        for _ in range(warmup):
            execute()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        timings = np.empty(repetitions, dtype=np.float64)
        for index in range(repetitions):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter_ns()
            execute()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings[index] = (time.perf_counter_ns() - start) / 1.0e6
        field_timings = np.empty(repetitions, dtype=np.float64)
        for index in range(repetitions):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter_ns()
            execute_field_only()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            field_timings[index] = (time.perf_counter_ns() - start) / 1.0e6
        full_mean = float(np.mean(timings))
        field_mean = float(np.mean(field_timings))
        rows.append(
            {
                "query_count": int(query_count),
                "latency_median_ms": float(np.median(timings)),
                "latency_p95_ms": float(np.quantile(timings, 0.95)),
                "throughput_queries_per_second": float(
                    1000.0 * query_count / full_mean
                ),
                "field_query_median_ms": float(np.median(field_timings)),
                "field_query_p95_ms": float(np.quantile(field_timings, 0.95)),
                "field_query_compute_fraction": float(
                    min(1.0, field_mean / full_mean)
                )
                if full_mean > 0
                else float("nan"),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0,
                "cached_coordinates": True,
                "text_encoding_included": False,
            }
        )
    return rows
