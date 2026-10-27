"""Source and artifact fingerprints for reproducible experiments."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_fingerprint() -> dict[str, str]:
    """Hash executable sources and protocol inputs, including uncommitted files."""
    paths: set[Path] = set()
    for directory in ("src", "rtl", "sim", "scripts", "tests", "configs", "viewer", ".github"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                paths.add(path)
    for name in (
        "Makefile",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "docs/ARCHITECTURE.md",
        "docs/EXPERIMENTS.md",
    ):
        paths.add(ROOT / name)
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def environment() -> dict[str, Any]:
    def command(args: list[str]) -> str:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
        return result.stdout.strip()

    return {
        "source_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--porcelain"]),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "verilator": command(["verilator", "--version"]),
        "compiler": command(["c++", "--version"]).splitlines()[0],
        "uv": command(["uv", "--version"]),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scikit-learn", "scipy", "pytest")
        },
    }
