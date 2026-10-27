"""Host orchestration and explicit transfer accounting for the RTL datapath."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tinynpu.model import QuantizedModel
from tinynpu.reference import matmul_int8
from tinynpu.simulator import Simulator


@dataclass
class HardwareResult:
    layer1_acc: NDArray[np.int32]
    layer2_acc: NDArray[np.int32]
    logits: NDArray[np.float64]
    predictions: NDArray[np.int64]
    tiles: list[dict[str, int]]
    traces: list[dict[str, Any]]


class InferenceMismatch(AssertionError):
    """A failed hardware gate with enough evidence to reproduce its tile."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def transfer_cost(m: int, k: int, n: int, bandwidth: int) -> dict[str, int]:
    """Reload each tile, synthesize padding locally, serialize transfers."""
    if not (1 <= m <= 4 and 1 <= n <= 4 and 1 <= k <= 64):
        raise ValueError("Tile dimensions are outside the fixed architecture.")
    if not isinstance(bandwidth, int) or isinstance(bandwidth, bool) or bandwidth <= 0:
        raise ValueError("Bandwidth must be a positive integer in bytes/cycle.")
    load_bytes = (m + n) * k
    store_bytes = 4 * m * n
    return {
        "load_bytes": load_bytes,
        "store_bytes": store_bytes,
        "load_cycles": (load_bytes + bandwidth - 1) // bandwidth,
        "store_cycles": (store_bytes + bandwidth - 1) // bandwidth,
    }


def infer_hardware(
    model: QuantizedModel,
    x: NDArray[np.float64],
    simulator: Simulator,
    *,
    trace: bool = False,
    job_id: int = 0,
) -> HardwareResult:
    """Run both layers through RTL; verify every tile before using its output."""
    if x.ndim != 2 or x.shape[1] != 64 or not 1 <= len(x) <= 4:
        raise ValueError("Hardware inference requires 1..4 images with 64 features.")
    tiles: list[dict[str, int]] = []
    traces: list[dict[str, Any]] = []

    def layer(a: NDArray[np.int8], weights: NDArray[np.int8], layer_id: int) -> NDArray[np.int32]:
        result = np.empty((len(a), weights.shape[1]), dtype=np.int32)
        for start in range(0, weights.shape[1], 4):
            b = weights[:, start : start + 4]
            tile_id = len(tiles)
            tile = simulator.run(a, b, trace=trace, job_id=job_id, tile_id=tile_id)
            expected = matmul_int8(a, b)
            diagnostics = {
                "job_id": job_id,
                "layer": layer_id,
                "tile_id": tile_id,
                "a": a.tolist(),
                "b": b.tolist(),
                "expected": expected.tolist(),
                "actual": tile.output.tolist(),
                "counters": {
                    key: getattr(tile, key)
                    for key in (
                        "compute_cycles",
                        "pause_cycles",
                        "clear_cycles",
                        "valid_macs",
                        "useful_macs",
                        "padded_macs",
                    )
                },
            }
            if not np.array_equal(tile.output, expected):
                raise InferenceMismatch(
                    f"RTL mismatch: job {job_id}, layer {layer_id}, tile {tile_id}", diagnostics
                )
            m, k, n = len(a), a.shape[1], b.shape[1]
            if (
                tile.compute_cycles != k + 6
                or tile.clear_cycles != 1
                or tile.pause_cycles != 0
                or tile.valid_macs != 16 * k
                or tile.useful_macs != m * k * n
                or tile.padded_macs != 16 * k - m * k * n
            ):
                raise InferenceMismatch(
                    f"RTL counter mismatch: job {job_id}, tile {tile_id}", diagnostics
                )
            result[:, start : start + n] = tile.output
            tiles.append(
                {
                    "job_id": job_id,
                    "tile_id": tile_id,
                    "layer": layer_id,
                    "output_start": start,
                    "m": m,
                    "k": k,
                    "n": n,
                    "compute_cycles": tile.compute_cycles,
                    "pause_cycles": tile.pause_cycles,
                    "clear_cycles": tile.clear_cycles,
                    "valid_macs": tile.valid_macs,
                    "useful_macs": tile.useful_macs,
                    "padded_macs": tile.padded_macs,
                }
            )
            for cycle in tile.trace:
                traces.append({**cycle, "layer": layer_id, "output_start": start})
        return result

    acc1 = layer(model.quantize_input(x), model.w1, 1)
    acc2 = layer(model.hidden_from_acc(acc1), model.w2, 2)
    logits = model.logits_from_acc(acc2)
    return HardwareResult(acc1, acc2, logits, np.argmax(logits, axis=1), tiles, traces)
