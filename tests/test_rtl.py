from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from tinynpu.reference import matmul_int8
from tinynpu.simulator import Simulator, build_simulator


@pytest.fixture(scope="module")
def rtl_binary() -> Path:
    return build_simulator()


@pytest.fixture(scope="module")
def simulator(rtl_binary: Path) -> Iterator[Simulator]:
    with Simulator(rtl_binary) as instance:
        yield instance


def test_1024_seeded_tiles_against_independent_oracle(simulator: Simulator) -> None:
    rng = np.random.default_rng(20261027)
    count = 0
    for k in (1, 4, 16, 64):
        for m in range(1, 5):
            for n in range(1, 5):
                for repeat in range(16):
                    a = rng.integers(-128, 128, (m, k), dtype=np.int8)
                    b = rng.integers(-128, 128, (k, n), dtype=np.int8)
                    pauses = (0, k // 2, k + 5) if k > 1 and repeat % 2 else ()
                    actual = simulator.run(a, b, pause_steps=pauses)
                    np.testing.assert_array_equal(
                        actual.output, matmul_int8(a, b), err_msg=f"seed=20261027 case={count}"
                    )
                    assert actual.output.dtype == np.int32
                    assert actual.compute_cycles == k + 6
                    assert actual.clear_cycles == 1
                    assert actual.pause_cycles == len(pauses)
                    assert actual.valid_macs == 16 * k
                    assert actual.useful_macs == m * n * k
                    assert actual.padded_macs == (16 - m * n) * k
                    assert actual.trace == []
                    count += 1
    assert count == 1024


@pytest.mark.parametrize(("left", "right"), [(-128, -128), (-128, 127), (127, 127), (0, 127)])
def test_extremes_and_zero_values_are_useful(simulator: Simulator, left: int, right: int) -> None:
    a = np.full((4, 64), left, dtype=np.int8)
    b = np.full((64, 4), right, dtype=np.int8)
    result = simulator.run(a, b)
    np.testing.assert_array_equal(result.output, np.full((4, 4), 64 * left * right))
    assert result.useful_macs == 1024
    assert result.padded_macs == 0


def test_trace_reconciles_consumed_operands_accumulators_and_counters(simulator: Simulator) -> None:
    a = np.array([[1, -2], [-3, 4]], dtype=np.int8)
    b = np.array([[5, -6, 0], [-7, 8, 1]], dtype=np.int8)
    result = simulator.run(a, b, trace=True, pause_steps=(0, 3, 7), job_id=12, tile_id=3)
    np.testing.assert_array_equal(result.output, [[19, -22, -2], [-43, 50, 4]])
    assert len(result.trace) == result.compute_cycles + result.pause_cycles + result.clear_cycles
    prior = np.zeros((4, 4), dtype=np.int32)
    total_active = total_useful = total_padded = 0
    for cycle, record in enumerate(result.trace):
        assert record["cycle"] == cycle
        assert record["schema_version"] == 1
        assert record["job_id"] == 12 and record["tile_id"] == 3
        for i, row in enumerate(record["pes"]):
            for j, pe in enumerate(row):
                active = record["enable"] and pe["a_valid"] and pe["b_valid"]
                assert pe["active"] == active
                assert pe["useful"] == (active and i < 2 and j < 3)
                assert pe["padded"] == (active and not pe["useful"])
                expected = int(prior[i, j]) + (pe["a"] * pe["b"] if active else 0)
                assert pe["accumulator"] == expected
                prior[i, j] = pe["accumulator"]
                total_active += pe["active"]
                total_useful += pe["useful"]
                total_padded += pe["padded"]
                if active:
                    t = record["logical_step"] - i - j
                    assert 0 <= t < 2
                    assert pe["a"] == (int(a[i, t]) if i < 2 else 0)
                    assert pe["b"] == (int(b[t, j]) if j < 3 else 0)
    assert (total_active, total_useful, total_padded) == (
        result.valid_macs,
        result.useful_macs,
        result.padded_macs,
    )
    np.testing.assert_array_equal(prior[:2, :3], result.output)
    assert np.count_nonzero(prior[2:, :]) == 0
    assert np.count_nonzero(prior[:, 3:]) == 0
    # Last PE consumes its final product on the documented last enabled edge.
    assert result.trace[-1]["logical_step"] == 7
    assert result.trace[-1]["pes"][3][3]["active"]


def test_back_to_back_clear_discards_prior_accumulators(simulator: Simulator) -> None:
    simulator.run(np.full((4, 64), -128, np.int8), np.full((64, 4), -128, np.int8))
    result = simulator.run(np.zeros((1, 1), np.int8), np.zeros((1, 1), np.int8), trace=True)
    assert result.trace[0]["phase"] == "clear"
    assert result.trace[0]["logical_step"] == -1
    assert not result.trace[0]["enable"]
    assert all(pe["accumulator"] == 0 for row in result.trace[0]["pes"] for pe in row)
    np.testing.assert_array_equal(result.output, [[0]])


@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    [((5, 4), (4, 1)), ((1, 4), (4, 5)), ((1, 65), (65, 1)), ((4,), (4, 1))],
)
def test_invalid_shapes(simulator: Simulator, a_shape: tuple, b_shape: tuple) -> None:
    with pytest.raises(ValueError):
        simulator.run(np.zeros(a_shape, np.int8), np.zeros(b_shape, np.int8))


def test_bad_dtype_and_pauses(simulator: Simulator) -> None:
    a = np.ones((1, 1), dtype=np.int8)
    with pytest.raises(TypeError, match="int8"):
        simulator.run(a.astype(np.int16), a)
    for pauses in ((-1,), (7,), (0, 0), (True,)):
        with pytest.raises(ValueError, match="[Pp]ause"):
            simulator.run(a, a, pause_steps=pauses)


def test_closed_simulator_rejects_run(rtl_binary: Path) -> None:
    simulator = Simulator(rtl_binary)
    simulator.close()
    simulator.close()
    with pytest.raises(RuntimeError, match="closed"):
        simulator.run(np.ones((1, 1), np.int8), np.ones((1, 1), np.int8))


def test_every_reduction_length_with_every_step_paused(simulator: Simulator) -> None:
    rng = np.random.default_rng(20261028)
    for k in range(1, 65):
        m, n = 1 + k % 4, 1 + (k // 4) % 4
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b = rng.integers(-128, 128, (k, n), dtype=np.int8)
        result = simulator.run(a, b, trace=True, pause_steps=tuple(range(k + 6)))
        np.testing.assert_array_equal(result.output, matmul_int8(a, b))
        assert result.compute_cycles == sum(item["phase"] == "compute" for item in result.trace)
        assert result.pause_cycles == sum(item["phase"] == "pause" for item in result.trace)
        assert result.clear_cycles == sum(item["phase"] == "clear" for item in result.trace)
        assert result.compute_cycles == result.pause_cycles == k + 6
        assert result.clear_cycles == 1
        assert len(result.trace) == 2 * (k + 6) + 1
        observed_active = observed_useful = observed_padded = 0
        previous = np.zeros((4, 4), dtype=np.int32)
        for index, item in enumerate(result.trace):
            assert item["cycle"] == index
            for i, row in enumerate(item["pes"]):
                for j, pe in enumerate(row):
                    expected_active = item["enable"] and pe["a_valid"] and pe["b_valid"]
                    assert pe["active"] == expected_active
                    expected = int(previous[i, j])
                    if expected_active:
                        expected += pe["a"] * pe["b"]
                    assert pe["accumulator"] == expected
                    previous[i, j] = expected
                    observed_active += pe["active"]
                    observed_useful += pe["useful"]
                    observed_padded += pe["padded"]
        assert observed_active == result.valid_macs == 16 * k
        assert observed_useful == result.useful_macs == m * n * k
        assert observed_padded == result.padded_macs == (16 - m * n) * k
        np.testing.assert_array_equal(previous[:m, :n], result.output)
