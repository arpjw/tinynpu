# Two-day build plan

Budget: 14-16 focused hours after baseline setup. If HDL learning or debugging
takes longer, finish the verified array before adding the viewer.

## M0: Baseline setup

- Project brief, contracts, dependency lock, source directories, CI.
- Exact INT8 matrix-product oracle with sign and overflow checks.
- C++/Verilator toolchain fixture.
- Gate: make doctor and make check pass locally; hosted CI passes after push.

## Day 1

### M1: Processing element (hours 1-3)

Read ARCHITECTURE.md. Implement rtl/pe.sv and its simulation harness.
Test signed multiply, accumulation, forwarding, reset, clear, and global enable.

Gate: compare exact values to independent integer calculations. Exercise
-128, 127, zero, sign changes, invalid operands, and reset while disabled.

### M2: Array and schedule (hours 4-7)

Implement rtl/array.sv with 16 PEs and a host-side skewed input schedule.
Use K=1, 4, 16, 64. Start with a hand-calculated tile, then seeded random cases.
Add cycle traces from the simulated state and explicit job completion checks.

Gate: at least 1,000 seeded cases pass exactly. Cover padded tiles, global
pauses, repeated jobs, and clear between jobs. Verify K+6 enabled compute edges
under the documented convention. Save seeds and first-failure diagnostics.

### M3a: Software model (hour 8)

Create deterministic stratified train/calibration/test splits and persist
indices. Train the 64→16→10 classifier using only training data.
Export weights, scales, and quantized software predictions.

Gate: report floating-point and quantized software accuracy separately.
No target accuracy or quantization loss is assumed achieved.

## Day 2

### M3b: RTL-backed inference (hours 1-2)

Map both layers to tiles. Host code performs bias, ReLU, and requantization.
Reuse one compiled simulator across many inputs instead of recompiling per tile.

Gate: every matrix result matches the oracle exactly and held-out predictions
match the quantized software pipeline. A matching label alone is insufficient.

### M4: Measurement (hours 3-4)

Implement counters, bandwidth accounting, and the fixed batch/bandwidth sweep
in EXPERIMENTS.md. Freeze actual run settings before the held-out run.

Gate: trace counters reconcile with matrix dimensions. RTL cycle measurements
and modeled transfers appear in separate fields. Batch results preserve
per-image outputs.

### M5: Demo and write-up (hours 5-8)

Create a static viewer with image selection, cycle stepping, PE activity,
accumulator values, predictions, and batch comparison. Replay real traces.
Write a concise results page with commands, artifacts, limits, and findings.

Gate: reproduce one inference from a fresh checkout and inspect the viewer in
a browser. Package model/split hashes, configs, raw CSV, and representative traces.

## Scope cuts

Keep a fixed array, no independent stalls, no hardware activation pipeline,
no instruction set, no FPGA board, no synthesis/physical-design requirement.
If time slips, reduce viewer polish first. Preserve exact verification.
