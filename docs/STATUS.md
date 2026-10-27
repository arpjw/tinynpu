# Project status

## Implemented

- Repository scaffolding and shared AGENTS.md / CLAUDE.md guidance.
- Architecture contract and two-day build plan.
- Proposed experiment settings and evidence requirements.
- Exact signed INT8 matrix-product oracle with INT32 results.
- Reference tests for signs, extremes, padding, and invalid inputs.
- Verilator register fixture and C++ toolchain smoke driver.
- Python dependency lock, Make targets, and hosted CI configuration.
- Signed processing element, fixed 4×4 array, and persistent Verilator driver.
- Actual consumed-operand and accumulator traces; counted rising-edge categories.
- Deterministic training/calibration and safe numeric model serialization.
- RTL-backed two-layer inference with exact per-tile comparisons.
- Frozen experiment runner, transfer accounting, and trace viewer.

## Validation

Local implementation checks passed on 2026-10-27:

- Python 3.12.13, uv 0.11.26, Verilator 5.052, Apple clang 21.0.0.
- Ruff lint and formatting checks passed.
- 55 Python tests passed, including 1,024 seeded RTL comparisons and every K=1..64.
- 10,124 direct PE state checks passed.
- Two-layer inference matched the reference, including batched versus single images.
- RTL lint and the compiled C++/Verilator smoke test passed.
- Frozen held-out inference: 3,150 tiles and 18,720 accumulator values matched.
- FP and quantized test accuracy: 345/360 each, with two differing predictions.
- Independent artifact/source/trace audit passed; see AUDIT.md.
- Browser controls and missing-data behavior passed; see RESULTS.md.

The CI result attached to the current commit is the hosted source of truth.
Tool versions can be inspected with make doctor.

## Milestones

M0 through M5 have passed local acceptance gates. The implementation is delivered
on the feature branch for review; no merge is performed automatically.
Hosted CI at the current commit remains the source of truth for Linux validation.
It runs the full test suite and frozen inference experiment from a fresh checkout.

Future hardware work such as SRAM control, synthesis, FPGA deployment, and
physical timing/power is outside this completed scope.
