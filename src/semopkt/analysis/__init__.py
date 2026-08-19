"""Aggregation and paper-output generation from raw run artifacts."""

from .aggregate import aggregate_runs, load_predictions
from .tables import generate_tables

__all__ = ["aggregate_runs", "generate_tables", "load_predictions"]

