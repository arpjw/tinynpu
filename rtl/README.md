# RTL implementation boundary

`pe.sv` implements a signed INT8 multiplier, INT32 accumulator, and registered
operand forwarding. `array.sv` connects a fixed 4x4 output-stationary array.
Read docs/ARCHITECTURE.md for the arithmetic and timing contract and
docs/SIMULATOR.md for simulation, trace semantics, and verification commands.

The fixture under tests/fixtures/ verifies only the simulation toolchain.
Accelerator correctness is checked separately with the direct PE harness
and randomized array comparisons against the independent Python oracle.
