# Trace viewer

A dependency-free viewer for recorded TinyNPU RTL traces. Serve the repository
root over HTTP, then open `/viewer/`. The default data source is
`/artifacts/run/demo.json`. Override it with a same-origin URL, for example
`/viewer/?data=../artifacts/another-run/demo.json`.

The viewer never synthesizes activity or substitutes mock data. Loading errors
are displayed explicitly. Playback is a visual inspection speed, unrelated to
hardware clock frequency. Native buttons, selects, and the range slider support
keyboard navigation; each processing element has an accessible state label.

## Demo schema v1

- `schema_version`: `1`.
- `summary`: `float_accuracy`, `quantized_accuracy` (fractions), `test_count`,
  `rtl_mismatches`, and `source_id`.
- `comparisons`: rows with `batch_size`, `bandwidth` (bytes/cycle),
  `compute_cycles_per_image`, `total_cycles_per_image`, `compute_utilization`,
  and `total_utilization` (fractions).
- `examples`: `id`, `label`, `prediction`, `pixels` (64 values from 0 to 16),
  `batch_size` (1 or 4), `batch_labels`, `batch_predictions`, `logits`, and
  `traces`. The same `id` can appear once per batch size to retain selection
  when switching batches.
- Each trace follows architecture schema v1, with `layer` and `output_start`
  added by the inference runner. `pes` is a 4 by 4 array containing `a`, `b`,
  `a_valid`, `b_valid`, `accumulator`, `active`, `useful`, and `padded`.

Operands describe the inputs consumed at the recorded clock edge. Accumulators
describe state after that edge. An active padded MAC receives a separate color
from a useful MAC. Real zero-valued operands are not treated as padding.

The comparison table assumes the architecture contract's no-overlap transfer
model: reload both operands per tile, generate padding locally, store only
useful INT32 outputs. Host bias, ReLU, rescaling, and argmax are excluded.
