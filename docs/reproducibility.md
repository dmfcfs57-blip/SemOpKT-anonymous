# Reproducibility record

Each run directory contains:

- resolved configuration and its SHA-256 hash;
- source-tree hash and Git revision when available;
- dataset and split-manifest hashes;
- encoder identifiers, immutable revisions, pooling and normalization;
- software and hardware inventory;
- train, validation and test counts;
- epoch history, selected epoch and checkpoint digest;
- interaction-level predictions and metric JSON;
- per-student and per-concept metric tables;
- trainable/total parameter counts, timing and peak accelerator memory;
- anomaly status and a machine-readable completion marker.

Run directories are keyed by experiment, dataset, protocol, model, seed, and a
configuration hash. Existing completed runs are reused only after the source
tree, specification, processed table, immutable manifests, encoder matrix, and
all recorded output-artifact hashes are revalidated. Failed runs remain
recorded with their exception class and message.

Every aggregation, inference, postprocessing, and figure command first checks
that `run.json` and `complete.json` are identical and that the recorded artifact
set and every SHA-256 digest still match. Manifest creation and loading enforce
split hashes and leakage rules. The separate repository audit scans committed
text and Git history for identity-bearing or machine-specific material.
