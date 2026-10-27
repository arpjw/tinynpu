# Architecture contract v1

This specifies the planned implementation. See STATUS.md for current evidence.

## Compute

The array computes C = A @ B, where A is 4×K, B is K×4, and 1 <= K <= 64.
Operands are signed two's-complement INT8. Products are signed 16-bit values,
explicitly sign-extended into signed INT32 accumulators. No intermediate
saturation or truncation is permitted.

The maximum absolute sum is bounded by 64 * 128 * 128 = 1,048,576, so the initial
workload fits INT32. Supporting larger K requires revisiting that proof.
The independent Python oracle computes in INT64 and returns INT32.

Each processing element has:

- clk, synchronous active-high rst, clear, and global enable;
- signed a_in and b_in, each with its own valid input;
- registered a_out and b_out and corresponding valid outputs;
- a signed 32-bit accumulator.

Priority: reset, then clear, then enable. Reset and clear zero the accumulator,
forwarded operands, and valid bits even when enable is low. With enable low,
all state holds. On an enabled edge, forward both operands and validity bits;
accumulate only if both incoming valid bits are true.

All elements share enable. Independent lane backpressure is outside this scope.
The host advances the input schedule only on enabled compute edges.

## Dataflow and timing

Use output-stationary accumulation. A travels right; B travels down.
Each hop takes one enabled rising edge. At step t, inject:

- A[i,k] into row i when t = k + i;
- B[k,j] into column j when t = k + j.

Otherwise drive a zero operand with valid low. Operands meet at PE(i,j)
at step k + i + j. Step zero is the first enabled edge that multiplies
A[0,0] and B[0,0] in PE(0,0).

The last multiply happens at step K + 5, so a full tile takes K + 6 enabled
compute edges. The host reads after evaluating that final edge. A separate
clear edge precedes each tile. Memory loading and result extraction are
outside this count. Paused clock edges count toward elapsed cycles but do
not advance the logical compute step.

Baseline execution does not overlap jobs. It clears the array between tiles.
Back-to-back tests mean separate jobs with the required clear edge.

## Model mapping and host boundary

Use scikit-learn digits: 64 inputs, 16 ReLU hidden units, 10 outputs.
Rows represent images in a batch. Columns represent output neurons.

- First layer: four output-column tiles, each with K=64.
- Second layer: three output-column tiles, each with K=16.
- Batch 1 pads three rows. Batch 4 uses all four.
- The last output tile pads two columns.

The host trains the model, quantizes operands, schedules tiles, reads
accumulators, adds biases, applies ReLU, rescales, and chooses argmax.
The NPU performs matrix products only. Host operations are excluded from
RTL timing and must be disclosed in results.

Quantization plan: symmetric per-tensor scales for weights and activations,
round-to-nearest with ties-to-even, clipping to [-127, 127]. Hardware still
supports the full INT8 range and tests must cover -128.

For the first implementation, the host may dequantize INT32 results to floating
point, add the original bias, apply ReLU, and requantize for the next layer.
Reference and RTL-backed inference must use the same host operations.
This is not an all-integer end-to-end accelerator. Record scales and clipping
counts; fit activation scales using calibration data only.

## Counters and traces

Count useful MACs by valid, non-padding coordinates, including real zero-valued
operands. A zero numeric value is not automatically padding.

Track separately: clear edges, enabled compute edges, paused edges,
valid MACs, useful MACs, padded MACs, and modeled transfer cycles.
Record per-PE operand values, validity, accumulator, and useful/padded status.

Trace schema v1: one JSON object per cycle with schema_version, job_id, tile_id,
cycle, logical_step, phase, enable, and a 4×4 grid of PE states.
The viewer replays recorded traces; it must not fabricate activity from a
formula independently of the RTL.

## Memory model

Model load -> compute -> store with no overlap. This is a host-side accounting
model, not implemented SRAM or DMA hardware. An ideal local buffer supplies
up to four A bytes and four B bytes per enabled compute edge.

Conservative baseline: reload both operands for each output tile; do not retain
weights or reuse activations across tiles. Padding is generated locally.
Transfer only actual input rows and output columns. Store useful outputs as
INT32. State this policy in every result.

For m useful rows and n useful columns, load m*K + K*n bytes and store 4*m*n
bytes. At transfer rate B bytes/cycle, use ceil(load_bytes/B) and
ceil(store_bytes/B) separately. Total modeled tile cycles equal load cycles,
one clear edge, measured elapsed compute cycles, and store cycles.

This excludes host nonlinearities and real interconnect/bank contention.
Changing residency or overlap policy constitutes a separate experiment.

## Claims

RTL cycles describe the specified simulator. Transfer cycles describe the
chosen model. Neither establishes achieved clock frequency, wall-clock chip
latency, energy, silicon area, or GPU-relative speed.
