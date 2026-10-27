# RTL implementation boundary

No NPU modules exist yet. Start with pe.sv in M1, then array.sv in M2.
Use docs/ARCHITECTURE.md for the arithmetic and cycle contract.

The fixture under tests/fixtures/ verifies only the simulation toolchain.
Do not move it here or treat its passing test as accelerator correctness.
