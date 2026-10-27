# TinyNPU working agreements

Read README.md, docs/STATUS.md, docs/ARCHITECTURE.md, and nearby code before edits.
CLAUDE.md is a compatibility entry point; keep it intact.

- Preserve unrelated changes. Keep changes small and reviewable.
- Do not commit, push, deploy, or change repository visibility unless requested.
- Work from docs/BUILD_PLAN.md. Baseline setup is complete; NPU work starts at M1.
- Keep the fixed 4×4, INT8/INT32 scope unless the user changes it.
- SystemVerilog implements the datapath. Python controls tiling and host operations.
- The Python oracle must remain independent of RTL scheduling and implementation.
- Never silently narrow integer arithmetic. Test signs, extreme operands, reset,
  clear, enable, padding, and back-to-back jobs.
- Measure simulated cycles separately from modeled transfer costs and host time.
- Do not call modeled cycle counts physical latency, power, or silicon performance.
- Do not select quantization scales or tune models on the held-out test split.
- Use type hints and Ruff for Python. Use explicit signed widths in RTL.
- Run the narrow meaningful checks, then `make check` before a requested push.
- Generated weights, datasets, binaries, traces, and local results go in ignored
  artifacts/ or build/. Do not commit credentials or private environment files.
- Avoid em dashes in project prose.
- Update docs/STATUS.md when a milestone actually passes its acceptance gate.
