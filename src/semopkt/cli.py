"""Unified command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semopkt.analysis.aggregate import aggregate_runs
from semopkt.analysis.figures import generate_figures
from semopkt.analysis.inference import DEFAULT_COMPARISONS, system_comparison_inference
from semopkt.analysis.postprocess import postprocess_runs
from semopkt.analysis.tables import generate_tables
from semopkt.audit.anonymity import audit_anonymity
from semopkt.config import load_config
from semopkt.data.preprocess import preprocess_dataset
from semopkt.data.synthetic import generate_synthetic_interactions
from semopkt.experiments.catalog import build_experiment_specs
from semopkt.experiments.runner import ExperimentRunner
from semopkt.utils.io import write_json, write_table


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semopkt")
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--students", type=int, default=48)
    synthetic.add_argument("--concepts", type=int, default=12)
    synthetic.add_argument("--length", type=int, default=20)
    synthetic.add_argument("--seed", type=int, default=314159)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--raw-root", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--allow-statistic-mismatch", action="store_true")
    manifests = subparsers.add_parser("manifests")
    manifests.add_argument("--config", required=True)
    manifests.add_argument("--experiment", action="append")
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--experiment", required=True)
    run.add_argument("--model")
    run.add_argument("--dataset")
    run.add_argument("--seed", type=int)
    run.add_argument("--index", type=int)
    run.add_argument("--device")
    run.add_argument("--no-resume", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--runs", required=True)
    aggregate.add_argument("--output", required=True)
    outputs = subparsers.add_parser("outputs")
    outputs.add_argument("--runs", required=True)
    outputs.add_argument("--tables", required=True)
    outputs.add_argument("--figures", required=True)
    postprocess = subparsers.add_parser("postprocess")
    postprocess.add_argument("--runs", required=True)
    postprocess.add_argument("--output", required=True)
    inference = subparsers.add_parser("inference")
    inference.add_argument("--runs", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--models", nargs="*", default=list(DEFAULT_COMPARISONS))
    inference.add_argument("--reference", default="SemOpKT")
    inference.add_argument("--resamples", type=int, default=10000)
    inference.add_argument("--confidence", type=float, default=0.95)
    inference.add_argument("--seed", type=int, default=271828)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--root", default=".")
    audit.add_argument("--report")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "synthetic":
        generate_synthetic_interactions(
            arguments.output,
            students=arguments.students,
            concepts=arguments.concepts,
            sequence_length=arguments.length,
            seed=arguments.seed,
        )
        return 0
    if arguments.command == "prepare":
        preprocess_dataset(
            arguments.config,
            arguments.raw_root,
            arguments.output,
            allow_statistic_mismatch=arguments.allow_statistic_mismatch,
        )
        return 0
    if arguments.command == "manifests":
        config = load_config(arguments.config)
        runner = ExperimentRunner(config)
        experiments = arguments.experiment or list(config["experiments"])
        count = 0
        for experiment in experiments:
            for spec in build_experiment_specs(config, experiment):
                if spec.source_dataset is not None:
                    continue
                frame = runner._load_dataset(spec.dataset)
                runner._manifest(frame, spec)
                count += 1
        print(json.dumps({"manifests_checked": count}, sort_keys=True))
        return 0
    if arguments.command == "run":
        config = load_config(arguments.config)
        specs = build_experiment_specs(config, arguments.experiment)
        if arguments.model:
            specs = [spec for spec in specs if spec.model == arguments.model]
        if arguments.dataset:
            specs = [spec for spec in specs if spec.dataset == arguments.dataset]
        if arguments.seed is not None:
            specs = [spec for spec in specs if spec.seed == arguments.seed]
        if arguments.index is not None:
            specs = [specs[arguments.index]]
        runner = ExperimentRunner(config, device=arguments.device)
        for spec in specs:
            runner.run(spec, resume=not arguments.no_resume)
        print(json.dumps({"runs_processed": len(specs)}, sort_keys=True))
        return 0
    if arguments.command == "aggregate":
        frame = aggregate_runs(arguments.runs)
        write_table(frame, arguments.output)
        return 0
    if arguments.command == "outputs":
        products = generate_tables(arguments.runs, arguments.tables)
        products.update(postprocess_runs(arguments.runs, arguments.tables))
        figures = generate_figures(arguments.runs, arguments.figures)
        print(json.dumps({"tables": len(products), "figures": len(figures)}, sort_keys=True))
        return 0
    if arguments.command == "postprocess":
        products = postprocess_runs(arguments.runs, arguments.output)
        print(json.dumps({"products": len(products)}, sort_keys=True))
        return 0
    if arguments.command == "inference":
        frame = system_comparison_inference(
            arguments.runs,
            arguments.output,
            comparison_models=arguments.models,
            reference_model=arguments.reference,
            resamples=arguments.resamples,
            confidence=arguments.confidence,
            seed=arguments.seed,
        )
        print(json.dumps({"comparisons": len(frame)}, sort_keys=True))
        return 0
    if arguments.command == "audit":
        report = audit_anonymity(arguments.root)
        if arguments.report:
            write_json(arguments.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 2
    raise RuntimeError(f"Unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
