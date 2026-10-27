# TinyNPU

A weekend inference-chip design lab: a 4×4 INT8 systolic array, a tiny digit
classifier, and a cycle-by-cycle explanation of where inference time goes.

**Status: baseline setup.** The numerical oracle and toolchain smoke test work.
The NPU processing element, array, classifier, and viewer are not implemented.
Passing baseline CI does not establish NPU correctness or performance.

## Scope

- 16 signed INT8 multiply-accumulate units with INT32 accumulation.
- Output-stationary dataflow: activations move right, weights move down,
  partial sums remain inside each processing element.
- A 64 → 16 → 10 digit classifier, with bias, ReLU, and requantization on the host.
- Batch sizes 1 and 4; explicit padding and useful-operation accounting.
- RTL compute-cycle measurements, separately labeled modeled transfer costs.

## Quick start

Requires Git, uv, Make, a C++ compiler, and Verilator 5.020 or newer.
On macOS, install Command Line Tools if needed, then `brew install uv verilator`.
On Ubuntu, install `build-essential verilator` and uv.

```sh
git clone https://github.com/arpjw/tinynpu.git
cd tinynpu
make setup
make doctor
make check
```

Python 3.12.13 and Python dependencies are pinned by `.python-version` and
`uv.lock`. Verilator and the C++ compiler are system dependencies; `make doctor`
reports their versions. CI exercises Linux and local validation exercises macOS.

`make check` runs Ruff, Python reference tests, RTL smoke lint, and a compiled
C++/Verilator smoke simulation. The smoke fixture is a register, not an NPU.
`make format` formats Python sources.

If macOS marks virtual-environment `.pth` files hidden, Python can skip the
editable package registration. The symptom is `ModuleNotFoundError: tinynpu`
after a successful install. Inspect `ls -lO .venv/lib/python3.12/site-packages/*.pth`.
If those files have the hidden flag, clear it with
`chflags nohidden .venv/lib/python3.12/site-packages/*.pth` and rerun checks.

## Project map

| Path | Purpose |
| --- | --- |
| `docs/BUILD_PLAN.md` | Two-day milestones and acceptance gates |
| `docs/ARCHITECTURE.md` | Arithmetic, timing, dataflow, and host boundary |
| `docs/EXPERIMENTS.md` | Planned protocol and claim limits |
| `docs/STATUS.md` | Implemented versus planned work |
| `src/tinynpu/reference.py` | Independent exact matrix-product oracle |
| `rtl/` | Future synthesizable NPU modules |
| `sim/` | C++ harness and future simulation driver |
| `tests/` | Reference tests and toolchain-only RTL fixture |
| `configs/baseline.toml` | Proposed experiment settings |
| `artifacts/` | Ignored generated models, traces, and results |

Start the build at **M1** in the build plan. Keep the first working array fixed
at 4×4. Physical frequency, silicon area, power, FPGA integration, and a full
instruction set are outside this weekend scope.

## References

- [Verilator guide](https://verilator.org/guide/latest/overview.html)
- [Scikit-learn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [Gemmini architecture reference](https://github.com/ucb-bar/gemmini)

This is an educational architecture experiment, not a measured physical chip.
