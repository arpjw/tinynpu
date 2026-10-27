import subprocess

from tinynpu.simulator import build_simulator


def test_direct_pe_control_and_signed_arithmetic() -> None:
    binary = build_simulator(top="pe")
    completed = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "PASS: 10124 direct PE state checks; seed=20261027" in completed.stdout
