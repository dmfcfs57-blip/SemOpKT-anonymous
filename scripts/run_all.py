#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from semopkt.config import load_config
from semopkt.experiments.catalog import build_experiment_specs
from semopkt.experiments.runner import ExperimentRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--experiments", nargs="*")
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    experiments = arguments.experiments or list(config["experiments"])
    runner = ExperimentRunner(config, device=arguments.device)
    completed = 0
    for experiment in experiments:
        for specification in build_experiment_specs(config, experiment):
            runner.run(specification, resume=not arguments.no_resume)
            completed += 1
    print(json.dumps({"runs_processed": completed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

