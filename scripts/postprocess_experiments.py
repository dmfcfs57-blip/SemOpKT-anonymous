#!/usr/bin/env python3
from semopkt.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["postprocess", *__import__("sys").argv[1:]]))
