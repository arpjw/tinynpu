from dataclasses import replace

import numpy as np
import pytest

from tinynpu.inference import InferenceMismatch, infer_hardware
from tinynpu.model import QuantizedModel
from tinynpu.simulator import Simulator, build_simulator


def small_model() -> QuantizedModel:
    rng = np.random.default_rng(618)
    w1 = rng.integers(-12, 13, size=(64, 16), dtype=np.int8)
    w2 = rng.integers(-12, 13, size=(16, 10), dtype=np.int8)
    return QuantizedModel(
        w1=w1,
        w2=w2,
        b1=np.linspace(-0.3, 0.3, 16),
        b2=np.linspace(-0.2, 0.2, 10),
        input_scale=0.125,
        w1_scale=0.01,
        hidden_scale=0.02,
        w2_scale=0.01,
        float_w1=w1.astype(np.float64) * 0.01,
        float_w2=w2.astype(np.float64) * 0.01,
    )


def test_two_layers_batching_and_trace_match_reference() -> None:
    model = small_model()
    rng = np.random.default_rng(2001)
    x = rng.uniform(0, 16, (4, 64))
    reference = model.infer_software(x)
    with Simulator(build_simulator()) as simulator:
        result = infer_hardware(model, x, simulator, trace=True)
        singles = [infer_hardware(model, x[i : i + 1], simulator) for i in range(4)]
    np.testing.assert_array_equal(result.layer1_acc, reference.layer1_acc)
    np.testing.assert_array_equal(result.layer2_acc, reference.layer2_acc)
    np.testing.assert_array_equal(result.logits, reference.logits)
    np.testing.assert_array_equal(
        result.predictions, np.concatenate([r.predictions for r in singles])
    )
    assert sum(t["compute_cycles"] for t in result.tiles) == 346
    assert sum(t["clear_cycles"] for t in result.tiles) == 7
    assert sum(t["useful_macs"] for t in result.tiles) == 4 * 1184
    assert len(result.traces) == 353
    assert {t["layer"] for t in result.traces} == {1, 2}


def test_host_rejects_corrupted_tile_before_using_it() -> None:
    class BrokenSimulator:
        def __init__(self, simulator: Simulator) -> None:
            self.simulator = simulator

        def run(self, *args, **kwargs):
            result = self.simulator.run(*args, **kwargs)
            bad_output = result.output.copy()
            bad_output[0, 0] += 1
            return replace(result, output=bad_output)

    with Simulator(build_simulator()) as simulator:
        with pytest.raises(InferenceMismatch, match="RTL mismatch") as caught:
            infer_hardware(small_model(), np.zeros((1, 64)), BrokenSimulator(simulator))
    assert caught.value.diagnostics["actual"][0][0] == 1
    assert caught.value.diagnostics["expected"][0][0] == 0
    assert len(caught.value.diagnostics["a"][0]) == 64
