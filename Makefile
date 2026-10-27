.PHONY: setup doctor lint format test rtl-lint sim-smoke check

UV ?= uv
VERILATOR ?= verilator

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

sim-smoke:
	mkdir -p build/toolchain
	$(VERILATOR) --cc --exe --build -Wall --top-module toolchain_smoke \
		--Mdir "$(CURDIR)/build/toolchain" \
		"$(CURDIR)/tests/fixtures/toolchain_smoke.sv" "$(CURDIR)/sim/smoke_main.cpp"
	./build/toolchain/Vtoolchain_smoke

check: lint test rtl-lint sim-smoke
