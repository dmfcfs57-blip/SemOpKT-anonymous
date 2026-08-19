#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from semopkt.analysis.reproduction import reproduction_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=0.010)
    arguments = parser.parse_args()
    result = reproduction_audit(
        arguments.runs,
        arguments.reference,
        arguments.output,
        tolerance=arguments.tolerance,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
