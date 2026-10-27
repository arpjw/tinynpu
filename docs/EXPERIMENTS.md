# Planned experiment protocol v1

Status: proposed settings, not a frozen executed experiment.
configs/baseline.toml is currently a planning artifact; no runner consumes it.

Question: how much useful work does this 4×4 datapath perform for batch 1
versus batch 4, and how does an explicit transfer model change utilization?

## Data and quality

- Seed: 20261027.
- Dataset: sklearn digits, original 8×8 inputs.
- Split: stratified 70% training, 10% calibration, 20% held-out test.
- Persist exact indices and hashes; record the integer counts resulting from
  split rounding. Use a deterministic two-stage stratified split.
- Train only on training data. Fit activation scales on calibration data.
- Freeze model, split, quantization, and run configuration before test evaluation.
- Report FP accuracy, quantized reference accuracy, and RTL/reference mismatches.
- Do not tune on test results. No accuracy or speedup claim is pre-established.

## Correctness controls

1. Hand-computed and extreme signed arithmetic cases.
2. At least 1,000 seeded RTL cases across supported K and edge conditions.
3. Exact INT32 agreement at every classifier matrix-product boundary.
4. Identical predictions for quantized software and RTL-backed inference.
5. Batch 1 and batch 4 preserve individual outputs and ordering.

The baseline oracle tests are not a substitute for these RTL controls.

## Performance sweep

Batch sizes: 1 and 4. Modeled transfer rates: 1, 4, 8 bytes/cycle.
Use the same ordered held-out examples in each condition. Pad only the last
batch as needed and exclude padded images from useful-work counts.

Use ARCHITECTURE.md's reload-per-tile policy and no transfer/compute overlap.
Record actual workload bytes. Padding is synthesized locally.
No cache warmness, implicit residency, or unreported reuse assumptions.

Report:

- simulated clear, enabled compute, pause, and elapsed compute cycles;
- useful MACs, padded MACs, and useful outputs;
- modeled load/store bytes and cycles;
- useful MACs / (16 * measured compute cycles);
- useful MACs / (16 * modeled load-to-store cycles);
- batch completion cycles and cycles per useful image separately.

A network image requires 64*16 + 16*10 = 1,184 useful MACs.
With the fixed schedule and no pauses, seven tiles require
4*(64+6) + 3*(16+6) = 346 enabled compute cycles per batch.
This is an analytical check to test, not an observed RTL result.
Seven clear cycles and transfer costs are additional.

For full batches, the corresponding analytical compute utilization is
1,184/(16*346) for batch 1 and 4*1,184/(16*346) for batch 4.
Do not count padding as useful work or interpret this as measured silicon
throughput. Batching wait time and host operations are outside this model.

## Evidence

Future run manifest: source commit and dirty state, OS/tool versions, seed,
configuration hash, split/model/scale hashes, invocation, schema versions,
raw results, and verification counts. Preserve failures and failed claim gates.

Deterministic cycle counts do not require statistical confidence intervals.
Report test-set size and classification accuracy as finite-sample observations.
Simulator wall-clock runtime may be recorded for developer convenience but
is not chip latency.
