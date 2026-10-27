# Results

The fixed 4×4 RTL array ran both layers of the classifier on 360 held-out images
in batches of 1 and 4. Every hardware matrix result matched the integer reference.
The run completed on 2026-10-27 from source commit
`740842dbb99c9275b56fe852d773752acc31c930`.

Source fingerprint:
`e24d9fb21109c9e9166c130b6f654b815fb97d7a20386de0d26257c481c920e9`.
These results describe that frozen snapshot. The fingerprint includes viewer,
build, and protocol files as well as executable sources, so later UI or packaging
changes produce a different fingerprint even when the datapath is unchanged.
Raw local evidence is in artifacts/run/; fresh CI runs generate their own
manifests and platform-specific binaries. This report does not attest that a
later checkout has the historical fingerprint or independently passed its checks.

## Correctness and model quality

| Check | Observed result |
| --- | --- |
| Python/RTL integration suite | 55 tests passed |
| Direct PE checks | 10,124 passed |
| Persisted randomized array campaign | 1,024 cases passed |
| Additional reduction-length test | Every K=1..64 with every step paused |
| Training/calibration/test images | 1,257 / 180 / 360 |
| Floating-point test accuracy | 345/360, 95.83% |
| Quantized test accuracy | 345/360, 95.83% |
| RTL tile comparisons in inference | 3,150 exact matches |
| Checked accumulator values in inference | 18,720 |
| RTL versus quantized prediction mismatches | 0 |

Training converged in 64 L-BFGS iterations under the fixed protocol. Quantization
changed predictions on two images despite unchanged overall accuracy. One of
5,760 held-out hidden activation values clipped; no input values clipped.
These observations describe this finite test split and model, not general
quantization accuracy preservation.

## Compute and transfers

Each batch required 346 counted compute edges plus seven clear edges. With the
fixed row-to-image mapping, useful compute utilization rose from **21.39%** at
batch 1 to **85.55%** at batch 4.

| Batch | Modeled bytes/cycle | RTL compute cycles/image | Modeled total cycles/image | Modeled total useful utilization |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 346 | 1,945 | 3.80% |
| 1 | 4 | 346 | 751 | 9.85% |
| 1 | 8 | 346 | 552 | 13.41% |
| 4 | 1 | 86.5 | 792.25 | 9.34% |
| 4 | 4 | 86.5 | 264.25 | 28.00% |
| 4 | 8 | 86.5 | 176.25 | 41.99% |

Transfers use a serialized load/compute/store model: reload each tile, generate
padding locally, and feed the array from an ideal local buffer during compute.
They are modeled costs, not measured external-memory traffic on a chip.

This demonstrates two architecture lessons. Matching workload shape to the array
can greatly improve useful compute occupancy. Transfer costs can still dominate
even when the arithmetic units are busy.

Throughput and latency differ: at 8 modeled bytes/cycle, batch 4 completes in
705 modeled cycles versus 552 for one image. Its 176.25 cycles/image is an
amortized throughput figure. Time waiting for four requests is excluded.

## Evidence and demo

The runner froze model, split, settings, qualification, source, and an archived
executable before evaluation, then verified unchanged hashes at completion.
See [the independent audit](AUDIT.md) for separate reconciliation of all manifest
files, saved predictions, byte/cycle accounting, and retained trace transitions.

At the time of this run, the viewer was inspected in the in-app browser at a
463-pixel viewport with no document overflow. Digit selection, batch switching,
cycle stepping, slider, play/pause, reset, final-frame behavior, and missing-artifact
errors worked.
All four examples replay 353 actual RTL records each, including tile clears.
No substitute animation is generated when evidence is unavailable.

Reproduce from a fresh checkout:

```sh
make setup
make check
make experiment
make serve
```

Open http://127.0.0.1:8765/viewer/. Existing evidence is never overwritten;
choose a fresh RUN_DIR for another run.

## Limits

Bias, ReLU, requantization, and argmax run on the host and are excluded from the
cycle totals. There is no implemented SRAM controller, DMA engine, instruction
set, or physical chip. Frequency, area, power, real-time latency, and GPU-relative
performance remain unmeasured. The model is a small educational classifier.
