# SemOpKT anonymous reproducibility repository

This repository accompanies the anonymous manuscript *SemOpKT: A
Function-Valued Semantic State Model for Open-World Cold-Start Knowledge
Tracing*. It contains the data-processing, model, experiment, statistical, and
reporting code while intentionally excluding author names, affiliations,
e-mail addresses, machine-specific paths, private dataset copies, pretrained
weights, and submission identifiers.

## Reproducibility boundary

The code implements the sequential prediction rule, frozen semantic encoder,
inducing-point student field, response-conditioned rank-separated proposal,
gated update, baselines, fixed student and concept manifests, and experiments
E0--E20. Raw datasets must be obtained from their public providers under their
respective terms. Generated predictions and tables are never hard-coded.

This is a clean-room executable reference implementation of the equations and
protocols in the anonymous manuscript. It does not contain private raw data or
historical checkpoints. A result is attributable to this repository only when
its run record, manifest, encoder cache, checkpoint, prediction files, and
source-tree hashes validate together.

The predictive encoder is pinned to
`sentence-transformers/all-mpnet-base-v2` revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130`. Semantic-cluster manifests use
the independent `sentence-transformers/all-MiniLM-L6-v2` revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Both use mean pooling and
L2-normalized sentence embeddings. Model downloads remain in an external
cache and are not committed.

## Installation

```bash
conda env create -f environment.yml
conda activate semopkt-anonymous
python -m pip install -e .
```

A CPU-only installation can instead install a compatible CPU PyTorch wheel
before `python -m pip install -e .`.

## Minimal verified workflow

```bash
python scripts/make_synthetic_data.py --output data/processed/Synthetic/interactions.csv
python scripts/make_splits.py --config configs/experiment/smoke.yaml
python scripts/run_experiment.py --config configs/experiment/smoke.yaml --experiment E1
python scripts/aggregate_results.py --runs runs --output tables/generated/smoke_metrics.csv
python scripts/postprocess_experiments.py --runs runs --output tables/generated
python scripts/generate_paper_outputs.py --runs runs --tables tables/generated --figures figures/generated
python scripts/statistical_inference.py --runs runs --output tables/generated/smoke_inference.csv --models DKT --resamples 100
python scripts/audit_repository.py --root .
```

The synthetic workflow validates plumbing only and is never used for paper
claims. Real runs use the three dataset configurations under `configs/data/`.

## Full experiment workflow

1. Place raw files under `data/raw/<dataset>/` without committing them.
2. Run `scripts/prepare_data.py` with the corresponding data configuration.
3. Freeze manifests with `scripts/make_splits.py`; hashes are embedded in JSON.
4. Run an experiment configuration with `scripts/run_experiment.py`.
5. Aggregate predictions with `scripts/aggregate_results.py`.
6. Generate paired protocol products with `scripts/postprocess_experiments.py`.
7. Generate tables and figures with `scripts/generate_paper_outputs.py`.
8. Run the pre-registered paired student bootstrap with
   `scripts/statistical_inference.py`.
9. Re-run aggregation to enforce artifact and manifest provenance checks, then
   run the identity/path scan with `scripts/audit_repository.py`.

For E0, compare rerun cells against a separately supplied published-anchor CSV
with `scripts/reproduction_audit.py`; executable code contains no published or
manuscript result values.

Use `scripts/tune_models.py` before the paper runs when hyperparameters have
not yet been frozen. It writes every validation-only candidate and the selected
configuration; it never produces test predictions.

`configs/experiment/paper.yaml` enumerates E0--E20, the ten seeds, the nested
training sizes, baselines, holdout ratios, and directed transfer pairs. Runs
are resumable: a completion marker is accepted only after revalidating its
configuration, source tree, processed data, immutable manifests, encoder
matrix, and output artifacts.

Run a bounded subset without editing configuration files:

```bash
python scripts/run_experiment.py \
  --config configs/experiment/paper.yaml \
  --experiment E5 --dataset NIPS34 --model SemOpKT --seed 202601
```

The catalog includes system and early-user cold start, random and cluster
unseen concepts, double cold start, online personal adaptation, target-only,
source-only, source-to-target and multi-source transfer, ablations, semantic
controls, operator replacements, vocabulary expansion, propagation, history
perturbations, calibration, data efficiency, sensitivity, runtime, error
groups, and fixed-rule qualitative case selection.

## Verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests scripts
python scripts/audit_repository.py --root .
```

The tests cover preprocessing, immutable splits, strict prediction-before-
update order, manuscript-scale parameter count, history-only perturbations,
source-only calibration isolation, few-shot transfer, and anonymous release.

## Anonymous-release rules

- Do not add personal names, affiliations, e-mail addresses, ORCID values,
  private URLs, local usernames, drive-letter paths, or home-directory paths.
- Do not commit raw data, model caches, checkpoints, prediction files, or logs.
- Create public release commits only with a neutral account and neutral commit
  metadata if the venue exposes repository history.
- Run `python scripts/audit_repository.py --root .` immediately before upload.

Protocol definitions and command examples are documented in `docs/`.
