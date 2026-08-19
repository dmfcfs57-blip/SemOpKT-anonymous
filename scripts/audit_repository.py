#!/usr/bin/env python3
from semopkt.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["audit", *__import__("sys").argv[1:]]))

