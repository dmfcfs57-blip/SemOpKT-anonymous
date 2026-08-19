"""Static repository scan for identity-bearing or machine-specific material."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".csv",
    ".tex",
    ".sh",
}
IGNORED_PARTS = {".git", ".venv", "__pycache__", "runs", "generated", "raw", "embeddings"}

PATTERNS = {
    "email": re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])"),
    "orcid": re.compile(r"\b\d{4}-\d{4}-\d{4}-[\dX]{4}\b", re.IGNORECASE),
    "windows_absolute_path": re.compile(r"\b[A-Za-z]:[\\/][^\s'\"`]+"),
    "unix_home_path": re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/"),
    "private_ipv4": re.compile(r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)\d{1,3}\.\d{1,3}\b"),
    "submission_identifier": re.compile(r"\b[A-Z]{2,8}-D-\d{2}-\d{4,8}\b"),
}


def _git_authors(root: Path) -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "log", "--format=%an <%ae>"],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return sorted(set(line.strip() for line in completed.stdout.splitlines() if line.strip()))
    except (OSError, subprocess.SubprocessError):
        return []


def audit_anonymity(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    findings: list[dict[str, Any]] = []
    scanned = 0
    for path in sorted(repository.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(repository)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        scanned += 1
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            findings.append({"type": "non_utf8_text", "path": relative.as_posix()})
            continue
        for line_number, line in enumerate(lines, start=1):
            for name, pattern in PATTERNS.items():
                for match in pattern.finditer(line):
                    value = match.group(0)
                    if name == "email" and value.endswith("@example.invalid"):
                        continue
                    findings.append(
                        {
                            "type": name,
                            "path": relative.as_posix(),
                            "line": line_number,
                            "value": value,
                        }
                    )
    forbidden_files = []
    for pattern in (".env", "*.pem", "*.key", "*.p12", "*.ckpt", "*.safetensors"):
        forbidden_files.extend(
            path.relative_to(repository).as_posix()
            for path in repository.rglob(pattern)
            if ".git" not in path.parts
        )
    authors = _git_authors(repository)
    nonanonymous_authors = [
        author
        for author in authors
        if "anonymous" not in author.casefold() and "noreply" not in author.casefold()
    ]
    for author in nonanonymous_authors:
        findings.append({"type": "git_author", "value": author})
    for path in sorted(set(forbidden_files)):
        findings.append({"type": "forbidden_file", "path": path})
    return {
        "passed": not findings,
        "root_name": repository.name,
        "scanned_text_files": scanned,
        "git_commit_authors": authors,
        "findings": findings,
    }
