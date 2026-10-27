.PHONY: setup doctor lint format test rtl-lint sim-smoke sim-build check verify experiment serve

UV ?= uv
VERILATOR ?= verilator
RUN_DIR ?= artifacts/run
CONFIG ?= configs/baseline.toml
PORT ?= 8765
# Use checkout sources explicitly; macOS can hide editable-install .pth files.
export PYTHONPATH := $(CURDIR)/src

setup:
	$(UV) sync --frozen

doctor:
	$(UV) run --frozen python scripts/doctor.py

lint:
	$(UV) run --frozen ruff check .
	$(UV) run --frozen ruff format --check .

format:
	$(UV) run --frozen ruff format .

test:
	$(UV) run --frozen pytest

rtl-lint:
	$(VERILATOR) --lint-only -Wall --top-module toolchain_smoke tests/fixtures/toolchain_smoke.sv
	$(VERILATOR) --lint-only -Wall --top-module array rtl/pe.sv rtl/array.sv

sim-build:
	$(UV) run --frozen python -c "from tinynpu.simulator import build_simulator; build_simulator()"

verify: sim-build
	$(UV) run --frozen python scripts/verify_rtl.py --output artifacts/rtl-verification.json --cases 1024 --seed 20261027

experiment: sim-build
	$(UV) run --frozen python -m tinynpu.experiment --output "$(RUN_DIR)" --config "$(CONFIG)"

serve:
	$(UV) run --frozen python -m http.server $(PORT) --bind 127.0.0.1

sim-smoke:
	mkdir -p build/toolchain
	$(VERILATOR) --cc --exe --build -Wall --top-module toolchain_smoke \
		--Mdir "$(CURDIR)/build/toolchain" \
		"$(CURDIR)/tests/fixtures/toolchain_smoke.sv" "$(CURDIR)/sim/smoke_main.cpp"
	./build/toolchain/Vtoolchain_smoke

check: lint test rtl-lint sim-smoke
