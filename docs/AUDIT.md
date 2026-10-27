# Independent evidence audit

**Result: PASS.** Read-only audit of `artifacts/run` on 2026-10-27. This audit
recomputed artifact checksums, saved-result arithmetic, fixed-model predictions,
and recorded trace transitions. It did not retrain, tune, rebuild, or rerun the
RTL campaign. Only this report was written.

This is a historical audit of the snapshot identified below. Later viewer,
packaging, and release changes are not covered by its source-linkage check.

- Frozen source commit: `740842dbb99c9275b56fe852d773752acc31c930`.
- Source ID: `e24d9fb21109c9e9166c130b6f654b815fb97d7a20386de0d26257c481c920e9`.
- Evaluated executable SHA-256:
  `c0e5ac96d9fceed0e0717d1ddad8dab597f44514a7634ad8bc7b4377b2672497`.

## Integrity and qualification

At audit time, all 13 manifest-listed files matched their SHA-256 and byte counts,
and the run contained no unlisted evidence file other than the manifest itself.
All 34 frozen source/configuration files matched the checkout inspected then.
The source ID recomputed from that fingerprint matched the freeze, summary, and
manifest. Model, splits, training metadata, configuration, and qualification hashes
matched the freeze. The archived executable, build executable inspected then,
freeze hash, and qualification executable hash all agreed.

The qualification artifact records 1,024 passing array cases and 10,124 direct
processing-element state checks. Its source hashes match the frozen sources.
The project additionally reports 55 passing tests; this artifact audit did not
rerun that test suite or independently repeat qualification.

## Classification and accounting

The saved split contains 1,257 training, 180 calibration, and 360 held-out indices,
with complete coverage and no overlap. Training metadata reports convergence.
The freeze precedes held-out evaluation in the reviewed runner; preparation
metadata records no held-out evaluation.

Independent NumPy calculations from the exported weights reproduce both saved
software prediction columns. Both floating-point and quantized models classify
**345/360 images correctly, or 95.83%**. Their predictions differ on two images;
equal accuracy does not mean identical predictions. All 720 saved RTL predictions
across batches 1 and 4 match the quantized predictions and retain dataset order
and labels. Input clipping is zero; one held-out hidden value clips.

All 3,150 tile records and 1,350 batch/bandwidth records reconcile with the fixed
matrix shapes, padding masks, seven clear edges per batch, and per-phase transfer
rounding. All six aggregate comparisons reproduce from those raw records. Each
batch uses 346 compute cycles; useful compute utilization is 21.39% for batch 1
and 85.55% for batch 4.

| Transfer bytes/cycle | Batch 1 modeled cycles/image | Batch 4 modeled cycles/image |
| --- | ---: | ---: |
| 1 | 1,945 | 792.25 |
| 4 | 751 | 264.25 |
| 8 | 552 | 176.25 |

The runner records exact checks on 18,720 accumulator values across both batching
conditions. Full accumulator arrays are not retained for every image, so this
audit independently verifies the retained trace subset rather than repeating
all those RTL checks.

## Trace audit and limits

All four saved demo examples match the dataset, software logits, and predictions.
Every one of their 1,412 recorded cycles and 19,456 valid MAC events reconciles:
operand values and skewed arrival steps, validity/activity masks, padding,
accumulator transitions, clear behavior, and final matrix results. Demo summary
and comparison values match the corresponding evidence files.

These are RTL-simulation results with a serialized host-side load/compute/store
model and ideal local buffer. Bias, ReLU, rescaling, argmax, and batching wait time
are excluded from the cycle estimates. No clock frequency, physical latency,
power, area, FPGA result, or GPU comparison is established. This audit validates
internal evidence consistency and linkage to the frozen source snapshot; it is
not an independent silicon measurement or a complete second execution of the
experiment. Consult current CI results for checks of later revisions.
