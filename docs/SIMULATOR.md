# RTL simulation interface

`rtl/pe.sv` is the signed INT8/INT32 processing element. `rtl/array.sv`
connects 16 instances and exposes the actual operand wires at each PE for
observation. `sim/array_main.cpp` schedules operands and advances the compiled
Verilator model. It does not compute matrix products in C++.

Build from the repository root:

```sh
make sim-build
```

Equivalent Python build helper: `tinynpu.simulator.build_simulator()`.
The executable defaults to `build/array/Varray`. The helper also supports
`top="pe"` to build the direct PE verification harness.

```python
import numpy as np
from tinynpu.simulator import Simulator

with Simulator() as simulator:
    result = simulator.run(
        np.array([[1, -2]], dtype=np.int8),
        np.array([[5, -6], [-7, 8]], dtype=np.int8),
        trace=True,
        pause_steps=(0, 3),
        job_id=0,
        tile_id=0,
    )
    assert result.output.tolist() == [[19, -22]]
```

Inputs must have `int8` dtype, with shapes `(m,K)` and `(K,n)`;
`m,n` are 1..4 and `K` is 1..64. The harness pads rows and columns with
valid zero operands. Every tile therefore executes `16*K` valid MACs,
of which `m*n*K` are useful. Numerically zero real operands remain useful.
The caller must close the simulator or use its context manager. One process
handles repeated jobs with a clear edge between them.

A pause step adds one disabled clock edge immediately before the named
logical compute step. Pauses do not advance the schedule. Each step can
appear once; steps must lie in `0..K+5`.

`TileResult` contains an INT32 `output` array, `compute_cycles`,
`pause_cycles`, `clear_cycles`, `valid_macs`, `useful_macs`, `padded_macs`,
and `trace`. Enabled compute cycles are `K+6`; clear cycles are one.
Elapsed RTL cycles include compute, pause, and clear. Transfer modeling
is outside this simulator.

## Trace semantics

With `trace=True`, every clear, pause, and compute edge has a schema-v1
record defined in ARCHITECTURE.md. `cycle` is zero-based within the tile.
The clear edge has `logical_step=-1`; its PE activity is false.
Each PE's `a`, `b`, and valid bits are observed **before** the rising edge;
`accumulator` is observed **after** it. Active means both actual incoming
valid wires are high on an enabled compute edge. Useful and padded are
annotations of active operations using the requested output dimensions.
Paused and clear edges have no active/useful/padded operations.

The C++ harness derives counters from those same observed wires even when
trace serialization is disabled. The Python wrapper decodes results and
performs input validation; it never reconstructs PE state from a formula.

## Reproducible verification

```sh
uv run --frozen python scripts/verify_rtl.py --cases 1024 \
  --seed 20261027 --output artifacts/verification/rtl.json
uv run --frozen pytest tests/test_pe.py tests/test_rtl.py
```

The campaign cycles through every combination of `m,n=1..4` and
`K=1,4,16,64`, with seeded full-range INT8 operands, repeated jobs,
and pauses. It records source hashes, binary hash, seed, pass count,
and direct PE verification output. A failed case records both matrices,
expected result, pause schedule, and error before raising.

Pytest additionally verifies signed extremes, invalid operands, reset and
clear priority, state hold, trace-to-counter reconciliation, and every
traced accumulation transition. The direct PE harness checks 10,124 edges.
