# Project status

## Implemented baseline

- Repository scaffolding and shared AGENTS.md / CLAUDE.md guidance.
- Architecture contract and two-day build plan.
- Proposed experiment settings and evidence requirements.
- Exact signed INT8 matrix-product oracle with INT32 results.
- Reference tests for signs, extremes, padding, and invalid inputs.
- Verilator register fixture and C++ toolchain smoke driver.
- Python dependency lock, Make targets, and hosted CI configuration.

## Validation

Local baseline validation passed on 2026-10-27:

- Python 3.12.13, uv 0.11.26, Verilator 5.052, Apple clang 21.0.0.
- Ruff lint and formatting checks passed.
- 15 Python tests passed, including 100 seeded oracle comparisons.
- RTL smoke lint and the compiled C++/Verilator smoke test passed.

The CI result attached to the current commit is the hosted source of truth.
Tool versions can be inspected with make doctor. These checks validate baseline
infrastructure and the numerical oracle, not an implemented NPU.

## Not implemented

- M1 processing element and M2 array/scheduler.
- RTL verification campaign.
- Classifier training, calibration, and exported weights.
- RTL-backed inference.
- Memory-model runner, counters, and experiment results.
- Trace viewer.

The next implementation task is M1 in BUILD_PLAN.md.
