#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from semopkt.experiments.tuning import validation_search


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    arguments = parser.parse_args()
    selected = validation_search(
        arguments.config,
        arguments.dataset,
        arguments.model,
        arguments.seed,
        arguments.output,
        device=arguments.device,
    )
    print(json.dumps(selected, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
