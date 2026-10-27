"""Report the local toolchain without exposing environment variables."""

import importlib.metadata
import platform
import shutil
import subprocess
import sys


def main() -> int:
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    missing = []
    for executable in ("uv", "git", "make", "c++", "verilator"):
        path = shutil.which(executable)
        if path is None:
            missing.append(executable)
            print(f"{executable}: MISSING")
            continue
        result = subprocess.run([path, "--version"], capture_output=True, text=True, check=False)
        lines = (result.stdout + result.stderr).splitlines()
        print(f"{executable}: {lines[0] if lines else 'version unavailable'}")
        if result.returncode:
            missing.append(executable)
    for package in ("numpy", "scikit-learn", "pytest", "ruff"):
        print(f"{package}: {importlib.metadata.version(package)}")
    if missing:
        print(f"Toolchain unavailable: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
