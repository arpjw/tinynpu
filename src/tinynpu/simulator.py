"""Persistent Verilator subprocess interface for the fixed 4x4 RTL array."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self, TypedDict

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = REPO_ROOT / "build" / "array" / "Varray"


class PEState(TypedDict):
    a: int
    b: int
    a_valid: bool
    b_valid: bool
    accumulator: int
    active: bool
    useful: bool
    padded: bool


class CycleTrace(TypedDict):
    schema_version: int
    job_id: int
    tile_id: int
    cycle: int
    logical_step: int
    phase: str
    enable: bool
    pes: list[list[PEState]]


@dataclass(frozen=True)
class TileResult:
    output: NDArray[np.int32]
    compute_cycles: int
    pause_cycles: int
    clear_cycles: int
    valid_macs: int
    useful_macs: int
    padded_macs: int
    trace: list[CycleTrace]


def build_simulator(binary_dir: Path | None = None, *, top: str = "array") -> Path:
    """Build the real RTL simulator; rebuilding is incremental under Verilator."""
    if top not in {"array", "pe"}:
        raise ValueError("Supported simulator tops are array and pe.")
    output = (binary_dir or REPO_ROOT / "build" / top).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [REPO_ROOT / "rtl" / "pe.sv"]
    if top == "array":
        sources.append(REPO_ROOT / "rtl" / "array.sv")
    subprocess.run(
        [
            os.environ.get("VERILATOR", "verilator"),
            "--cc",
            "--exe",
            "--build",
            "-Wall",
            "--top-module",
            top,
            "--Mdir",
            str(output),
            *(str(path) for path in sources),
            str(REPO_ROOT / "sim" / f"{top}_main.cpp"),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    return output / f"V{top}"


class Simulator:
    """Run independent tiles through one compiled, persistent RTL simulation.

    Padding is zero-valued and valid: all 16 PEs execute K MACs, while the useful
    coordinate mask distinguishes padding. Each pause step adds one disabled
    edge immediately before the corresponding logical compute step.
    """

    def __init__(self, binary: Path | None = None) -> None:
        executable = (binary or DEFAULT_BINARY).resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"Simulator missing: {executable}. Run make sim-build first.")
        self._process = subprocess.Popen(
            [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Shut down the subprocess and release its pipes (safe to call twice)."""
        if self._process.poll() is None:
            assert self._process.stdin is not None
            try:
                self._process.stdin.write("QUIT\n")
                self._process.stdin.flush()
            except BrokenPipeError:
                pass
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        for pipe in (self._process.stdin, self._process.stdout):
            if pipe is not None:
                pipe.close()

    def run(
        self,
        a: NDArray[np.int8],
        b: NDArray[np.int8],
        *,
        trace: bool = False,
        pause_steps: tuple[int, ...] = (),
        job_id: int = 0,
        tile_id: int = 0,
    ) -> TileResult:
        if a.dtype != np.int8 or b.dtype != np.int8:
            raise TypeError("Both operands must have dtype int8; implicit narrowing is forbidden.")
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Both operands must be two-dimensional matrices.")
        m, k = a.shape
        if b.shape[0] != k or not 1 <= k <= 64 or not 1 <= m <= 4 or not 1 <= b.shape[1] <= 4:
            raise ValueError("Require A(m,K), B(K,n) with m,n in 1..4 and K in 1..64.")
        if any(type(step) is not int or not 0 <= step < k + 6 for step in pause_steps):
            raise ValueError("Pause steps must be integers in 0..K+5.")
        if len(set(pause_steps)) != len(pause_steps):
            raise ValueError("Duplicate pause steps are not supported.")
        if any(
            type(identifier) is not int or not 0 <= identifier < 2**31
            for identifier in (job_id, tile_id)
        ):
            raise ValueError("Job and tile identifiers must be nonnegative signed INT32 integers.")
        if self._process.poll() is not None:
            raise RuntimeError("Simulator is closed or terminated.")
        fields = [m, b.shape[1], k, int(trace), job_id, tile_id, len(pause_steps), *pause_steps]
        request = " ".join(str(value) for value in (*fields, *a.flat, *b.flat)) + "\n"
        assert self._process.stdin is not None and self._process.stdout is not None
        try:
            self._process.stdin.write(request)
            self._process.stdin.flush()
            response = self._process.stdout.readline()
        except BrokenPipeError as error:
            raise RuntimeError("RTL simulator terminated while handling a tile.") from error
        if not response:
            raise RuntimeError("RTL simulator exited without a tile response.")
        result = json.loads(response)
        if "error" in result:
            raise RuntimeError(f"RTL harness rejected request: {result['error']}")
        result["output"] = np.asarray(result["output"], dtype=np.int32)
        return TileResult(**result)
