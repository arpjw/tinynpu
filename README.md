# TinyNPU

A weekend inference-chip design lab: a 4×4 INT8 systolic array, a tiny digit
classifier, and a cycle-by-cycle explanation of where inference time goes.

[Open the trace viewer](https://tinynpu.vercel.app) ·
[Architecture](docs/ARCHITECTURE.md) · [Results](docs/RESULTS.md)

The RTL datapath, persistent simulator, quantized classifier pipeline, experiment
runner, and trace viewer are implemented. See [project status](docs/STATUS.md)
for completed validation gates and [results](docs/RESULTS.md) for measured outcomes.

## Scope

- 16 signed INT8 multiply-accumulate units with INT32 accumulation.
- Output-stationary dataflow: activations move right, weights move down,
  partial sums remain inside each processing element.
- A 64 → 16 → 10 digit classifier, with bias, ReLU, and requantization on the host.
- Batch sizes 1 and 4; explicit padding and useful-operation accounting.
- RTL compute-cycle measurements, separately labeled modeled transfer costs.

## What you learn

- How operand skew and forwarding make a systolic array compute a matrix product.
- Why batch size and padding affect the fraction of useful MACs.
- How INT8 quantization and INT32 accumulation map a trained model onto hardware.
- Why moving operands can cost more cycles than computing with them.

Every matrix tile is checked against an independent integer reference. The viewer
connects those checks to the operands and accumulators at each simulated edge.

## Quick start

Requires Git, uv, Make, a C++ compiler, and Verilator 5.020 or newer.
On macOS, install Command Line Tools if needed, then `brew install uv verilator`.
On Ubuntu, install `build-essential verilator` and uv.

```sh
git clone https://github.com/arpjw/tinynpu.git
cd tinynpu
make setup
make doctor
make check
make experiment
make serve
```

Open [the local viewer](http://127.0.0.1:8765/viewer/) after starting the server.
Choose a digit and batch size, step through recorded RTL states, or press Play.
The page shows real accumulator traces and separates measured compute cycles
from modeled transfer costs.

The hosted viewer replays saved examples. Run the experiment locally to train
the fixed classifier, execute the RTL, and generate your own evidence.

Python 3.12.13 and Python dependencies are pinned by `.python-version` and
`uv.lock`. Verilator and the C++ compiler are system dependencies; `make doctor`
reports their versions. CI exercises Linux and local validation exercises macOS.

`make check` runs Ruff, reference/model/inference tests, real RTL verification,
RTL lint, and a toolchain smoke test. `make verify` separately saves a 1,024-case
RTL campaign. `make experiment` verifies hardware, prepares and freezes the model,
evaluates all held-out images in batches of 1 and 4, and writes artifacts/run/.
It refuses to overwrite evidence. Use `make experiment RUN_DIR=artifacts/run-2`
for a separate run. To view that run, append
`?data=../artifacts/run-2/demo.json` to the viewer URL.

`make format` formats Python sources. Make and pytest explicitly use checkout
sources so editable-install discovery cannot silently select an older package.

If macOS marks virtual-environment `.pth` files hidden, Python can skip the
editable package registration. The symptom is `ModuleNotFoundError: tinynpu`
after a successful install. Inspect `ls -lO .venv/lib/python3.12/site-packages/*.pth`.
If those files have the hidden flag, clear it with
`chflags nohidden .venv/lib/python3.12/site-packages/*.pth` and rerun checks.

## Project map

| Path | Purpose |
| --- | --- |
| `docs/BUILD_PLAN.md` | Two-day milestones and acceptance gates |
| `docs/ARCHITECTURE.md` | Arithmetic, timing, dataflow, and host boundary |
| `docs/EXPERIMENTS.md` | Fixed experiment protocol and claim limits |
| `docs/STATUS.md` | Implemented versus planned work |
| `docs/DEPLOYMENT.md` | Verified static export and Vercel deployment |
| `src/tinynpu/` | Oracle, model, simulator, inference, evidence, and experiment |
| `rtl/` | Synthesizable PE and fixed 4×4 array |
| `sim/` | Persistent C++ array driver and direct PE checks |
| `tests/` | Arithmetic, controls, traces, model isolation, and inference tests |
| `viewer/` | Static viewer replaying actual RTL traces |
| `configs/baseline.toml` | Enforced experiment settings |
| `artifacts/` | Ignored generated models, traces, and results |

The learning path follows M1 through M5 in the build plan. Inspect one PE,
then the array schedule, then model tiling, and finally transfer accounting.
Physical frequency, silicon area, power, FPGA integration, and a full instruction
set are outside this scope. Bias, ReLU, rescaling, and argmax remain host operations.

Each experiment includes a pre-evaluation freeze, immutable executable copy,
model/split/config hashes, raw CSVs, representative traces, and a file manifest.
CI runs the complete experiment from a fresh Linux checkout and preserves its
evidence as an Actions artifact for seven days. Access follows repository and
GitHub permissions. Generated artifacts are excluded from the source checkout.
Each fresh run records its own source fingerprint and toolchain; the historical
results document the exact snapshot that produced them. Rebuilt executable hashes
can differ across platforms and compiler versions.

## Contributing and licenses

See [CONTRIBUTING.md](CONTRIBUTING.md) for validation and contribution guidance.
Original project code and documentation use the [MIT license](LICENSE).
Digit images and labels originate from E. Alpaydin and C. Kaynak's UCI dataset,
licensed under CC BY 4.0. See [third-party notices](THIRD_PARTY_NOTICES.md) for
attribution, source links, and the distinction between code and data licensing.

## References

- [Verilator guide](https://verilator.org/guide/latest/overview.html)
- [Scikit-learn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [Gemmini architecture reference](https://github.com/ucb-bar/gemmini)

This is an educational architecture experiment, not a measured physical chip.
