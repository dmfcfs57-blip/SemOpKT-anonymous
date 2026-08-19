# Experiment catalog

Every experiment is expanded from `configs/experiment/paper.yaml`; no metric is
stored in the catalog. Use `--index` to dispatch one deterministic run from an
expanded experiment.

| ID | Scope | Dedicated products |
|---|---|---|
| E0 | preprocessing and baseline sanity | run audit and metrics |
| E1 | fixed-validation system cold start | predictions, seed summaries, paired inference |
| E2 | full development-training pool | predictions and metrics |
| E3 | early new-user history | history-window metrics |
| E4 | random unseen concepts | first-encounter and macro-concept metrics |
| E5 | independently encoded cluster holdout | first-encounter and distance covariates |
| E6 | double random/cluster cold start | disjoint-user first encounters |
| E7 | online unseen-concept adaptation | attempt groups and old-concept drift |
| E8 | target/source/multi-source transfer | pretrain/fine-tune checkpoints and de-duplication audit |
| E9 | A0--A14 hypothesis ablations | system, cluster and double-cluster metrics |
| E10 | descriptor controls | paired original/paraphrase drift |
| E11 | operator replacements | parameter, convergence and latency audit |
| E12 | vocabulary and inducing resolution | disjoint-user old/new metrics, query-only drift, graph-insertion drift, and independently fitted protocol drift |
| E13 | empirical/model propagation alignment | matrices, edge audit and alignment metrics |
| E14 | history-only perturbations | clean/perturbed pairs, drift, AUC change and sensitivity ratio |
| E15 | calibration and selective prediction | calibrated/uncalibrated outputs and curves |
| E16 | 2--128 student data efficiency | power-law fits and equivalent sample size |
| E17 | encoder, rank, inducing, layer, temperature and regularization sensitivity | two-dataset system screening plus three-dataset double-cluster confirmation and cost records |
| E18 | training and cached-query efficiency | median/P95 latency, memory and query scaling |
| E19 | pre-registered error groups | group metrics and high-confidence errors |
| E20 | fixed-rule qualitative selection | paired SemOpKT/CLST cases and per-target field-change traces |

```bash
python scripts/run_experiment.py \
  --config configs/experiment/paper.yaml \
  --experiment E13 --model SemOpKT --seed 202601

python scripts/run_all.py \
  --config configs/experiment/paper.yaml \
  --experiments E1 E4 E5 E6
```

E10 paraphrases, normalizations and definitions and E14 synonym substitutions
are immutable input resources described in `docs/text_resources.md`. A missing
resource is an error, so a control can never silently fall back to the original
descriptor.

E0 additionally requires an independently transcribed reference CSV with the
columns `dataset,model,train_size,reference_auc`. Place it under ignored
`data/reference/`, then run:

```bash
python scripts/reproduction_audit.py \
  --runs runs \
  --reference data/reference/clst_reference_auc.csv \
  --output tables/generated/E0_audit
```

The script emits every local-reference cell, per-dataset tables, the absolute
0.010 gate, a diagnostic issue list, and a difference figure. Reference values
remain data inputs and are never embedded in executable source.
