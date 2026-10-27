"""Persist a reproducible RTL verification campaign and first-failure diagnostics."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from tinynpu.reference import matmul_int8
from tinynpu.simulator import REPO_ROOT, Simulator, build_simulator


def verify_campaign(output: Path, cases: int = 1024, seed: int = 20261027) -> dict[str, object]:
    if cases < 1024:
        raise ValueError("The M2 gate requires at least 1024 cases.")
    binary = build_simulator()
    pe_binary = build_simulator(top="pe")
    pe_check = subprocess.run([str(pe_binary)], check=True, capture_output=True, text=True)
    sources = [
        "rtl/pe.sv",
        "rtl/array.sv",
        "sim/array_main.cpp",
        "sim/pe_main.cpp",
        "src/tinynpu/simulator.py",
        "src/tinynpu/reference.py",
        "scripts/verify_rtl.py",
    ]
    hashes = {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in sources}
    report: dict[str, object] = {
        "schema_version": 1,
        "seed": seed,
        "requested_cases": cases,
        "passed_cases": 0,
        "status": "running",
        "reduction_lengths": [1, 4, 16, 64],
        "rows_and_columns": [1, 2, 3, 4],
        "case_schedule": "cycle K,m,n across 64 shapes; all operands seeded random INT8",
        "source_sha256": hashes,
        "array_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "pe_check": pe_check.stdout.strip(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with Simulator(binary) as simulator:
        for index in range(cases):
            k = (1, 4, 16, 64)[index % 4]
            m = 1 + (index // 4) % 4
            n = 1 + (index // 16) % 4
            a = rng.integers(-128, 128, (m, k), dtype=np.int8)
            b = rng.integers(-128, 128, (k, n), dtype=np.int8)
            pauses = tuple(sorted({0, k // 2, k + 5})) if (index // 64) % 2 else ()
            expected = matmul_int8(a, b)
            try:
                result = simulator.run(a, b, pause_steps=pauses, job_id=index)
                np.testing.assert_array_equal(result.output, expected)
                assert result.compute_cycles == k + 6
                assert result.pause_cycles == len(pauses)
                assert result.clear_cycles == 1
                assert result.valid_macs == 16 * k
                assert result.useful_macs == m * n * k
                assert result.padded_macs == (16 - m * n) * k
            except Exception as error:
                report.update(
                    status="failed",
                    first_failure={
                        "case": index,
                        "a": a.tolist(),
                        "b": b.tolist(),
                        "pause_steps": pauses,
                        "expected": expected.tolist(),
                        "error": repr(error),
                    },
                )
                output.write_text(json.dumps(report, indent=2) + "\n")
                raise
            report["passed_cases"] = index + 1
    report["status"] = "passed"
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/verification/rtl.json"))
    parser.add_argument("--cases", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    report = verify_campaign(args.output, args.cases, args.seed)
    print(f"PASS: {report['passed_cases']} RTL tiles; evidence: {args.output}")
