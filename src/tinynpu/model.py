"""Deterministic digit model and shared host-side quantized inference operations.

Preparation never evaluates the held-out split. All public inputs use the raw
0..16 digit pixel scale; the training input normalization is folded into W1.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from tinynpu.reference import matmul_int8

SEED = 20261027
MODEL_SCHEMA_VERSION = 1
TRAINING_PARAMETERS = {
    "hidden_layer_sizes": (16,),
    "activation": "relu",
    "solver": "lbfgs",
    "alpha": 0.0001,
    "max_iter": 500,
    "max_fun": 15000,
    "tol": 0.0001,
}


def _finite_matrix(values: NDArray, columns: int, name: str) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != columns:
        raise ValueError(f"{name} must have nonempty shape (N, {columns}).")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _validate_scale(scale: float) -> None:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Quantization scale must be finite and strictly positive.")


def symmetric_scale(values: NDArray) -> float:
    """Calibrate a per-tensor absmax scale, using 1 for an all-zero tensor."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Scale calibration requires nonempty finite values.")
    maximum = float(np.max(np.abs(values)))
    scale = maximum / 127.0 if maximum else 1.0
    _validate_scale(scale)
    return scale


def quantize_symmetric(values: NDArray, scale: float) -> NDArray[np.int8]:
    """Round ties to even, then saturate to [-127, 127], without wrapping."""
    _validate_scale(scale)
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Quantization requires finite values.")
    # Clipping before division bounds arithmetic even for very large inputs.
    clipped = np.clip(values, -127.0 * scale, 127.0 * scale)
    return np.clip(np.rint(clipped / scale), -127, 127).astype(np.int8)


def clipping_count(values: NDArray, scale: float) -> int:
    """Count values beyond the representable range before clipping/rounding."""
    _validate_scale(scale)
    return int(np.count_nonzero(np.abs(values) > 127.0 * scale))


def make_splits(labels: NDArray, seed: int = SEED) -> dict[str, NDArray[np.int64]]:
    """Stratify off 20% test, then 12.5% of the remainder for calibration."""
    indices = np.arange(len(labels), dtype=np.int64)
    remaining, test = train_test_split(indices, test_size=0.20, random_state=seed, stratify=labels)
    train, calibration = train_test_split(
        remaining, test_size=0.125, random_state=seed, stratify=labels[remaining]
    )
    return {"train_indices": train, "calibration_indices": calibration, "test_indices": test}


@dataclass(frozen=True)
class SoftwareResult:
    logits: NDArray[np.float64]
    predictions: NDArray[np.int64]
    layer1_acc: NDArray[np.int32]
    layer2_acc: NDArray[np.int32]
    quantized_input: NDArray[np.int8]
    hidden_quantized: NDArray[np.int8]


@dataclass(frozen=True)
class QuantizedModel:
    w1: NDArray[np.int8]
    w2: NDArray[np.int8]
    b1: NDArray[np.float64]
    b2: NDArray[np.float64]
    input_scale: float
    w1_scale: float
    hidden_scale: float
    w2_scale: float
    float_w1: NDArray[np.float64]
    float_w2: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name, shape in (("w1", (64, 16)), ("w2", (16, 10))):
            value = getattr(self, name)
            if value.shape != shape or value.dtype != np.int8:
                raise ValueError(f"{name} must have shape {shape} and dtype int8.")
        for name, shape in (
            ("b1", (16,)),
            ("b2", (10,)),
            ("float_w1", (64, 16)),
            ("float_w2", (16, 10)),
        ):
            value = getattr(self, name)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must have shape {shape} and finite values.")
        for scale in (self.input_scale, self.w1_scale, self.hidden_scale, self.w2_scale):
            _validate_scale(scale)

    @property
    def float_weights(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.float_w1, self.float_w2

    def quantize_input(self, x: NDArray) -> NDArray[np.int8]:
        return quantize_symmetric(_finite_matrix(x, 64, "Input"), self.input_scale)

    def hidden_from_acc(self, acc: NDArray[np.int32]) -> NDArray[np.int8]:
        """Shared host bias, ReLU, and requantization after the first product."""
        return quantize_symmetric(self.hidden_float_from_acc(acc), self.hidden_scale)

    def hidden_float_from_acc(self, acc: NDArray[np.int32]) -> NDArray[np.float64]:
        if acc.dtype != np.int32:
            raise TypeError("Layer 1 accumulators must have dtype int32.")
        wide = _finite_matrix(acc, 16, "Layer 1 accumulator")
        return np.maximum(wide * (self.input_scale * self.w1_scale) + self.b1, 0.0)

    def logits_from_acc(self, acc: NDArray[np.int32]) -> NDArray[np.float64]:
        if acc.dtype != np.int32:
            raise TypeError("Layer 2 accumulators must have dtype int32.")
        wide = _finite_matrix(acc, 10, "Layer 2 accumulator")
        return wide * (self.hidden_scale * self.w2_scale) + self.b2

    def predict_float(self, x: NDArray) -> NDArray[np.int64]:
        x = _finite_matrix(x, 64, "Input")
        hidden = np.maximum(x @ self.float_w1 + self.b1, 0.0)
        return np.argmax(hidden @ self.float_w2 + self.b2, axis=1).astype(np.int64)

    def infer_software(self, x: NDArray) -> SoftwareResult:
        q_input = self.quantize_input(x)
        acc1 = matmul_int8(q_input, self.w1)
        q_hidden = self.hidden_from_acc(acc1)
        acc2 = matmul_int8(q_hidden, self.w2)
        logits = self.logits_from_acc(acc2)
        return SoftwareResult(logits, np.argmax(logits, axis=1), acc1, acc2, q_input, q_hidden)


def load_model(model_path: Path) -> QuantizedModel:
    """Load numeric arrays only, without allowing serialized Python objects."""
    with np.load(model_path, allow_pickle=False) as data:
        if int(data["schema_version"]) != MODEL_SCHEMA_VERSION:
            raise ValueError("Unsupported model schema version.")
        return QuantizedModel(
            **{name: data[name] for name in ("w1", "w2", "b1", "b2", "float_w1", "float_w2")},
            **{
                name: float(data[name])
                for name in ("input_scale", "w1_scale", "hidden_scale", "w2_scale")
            },
        )


def prepare_model(output_dir: Path, seed: int = SEED) -> dict[str, Path]:
    """Train and calibrate the frozen baseline; never evaluate held-out inputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y = load_digits(return_X_y=True)
    splits = make_splits(y, seed)
    train = splits["train_indices"]
    calibration = splits["calibration_indices"]
    classifier = MLPClassifier(**TRAINING_PARAMETERS, random_state=seed)
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(x[train] / 16.0, y[train])
    # This fixed normalization is independent of all data statistics.
    float_w1 = classifier.coefs_[0] / 16.0
    float_w2 = classifier.coefs_[1]
    b1, b2 = classifier.intercepts_
    input_scale = symmetric_scale(x[calibration])
    w1_scale, w2_scale = symmetric_scale(float_w1), symmetric_scale(float_w2)
    w1, w2 = quantize_symmetric(float_w1, w1_scale), quantize_symmetric(float_w2, w2_scale)
    q_calibration = quantize_symmetric(x[calibration], input_scale)
    cal_acc1 = matmul_int8(q_calibration, w1)
    cal_hidden = np.maximum(cal_acc1.astype(np.float64) * (input_scale * w1_scale) + b1, 0.0)
    scales = {
        "input_scale": input_scale,
        "w1_scale": w1_scale,
        "hidden_scale": symmetric_scale(cal_hidden),
        "w2_scale": w2_scale,
    }
    paths = {
        name: output_dir / filename
        for name, filename in (
            ("model", "model.npz"),
            ("splits", "splits.npz"),
            ("training", "training.json"),
        )
    }
    np.savez_compressed(
        paths["model"],
        schema_version=MODEL_SCHEMA_VERSION,
        w1=w1,
        w2=w2,
        b1=b1,
        b2=b2,
        float_w1=float_w1,
        float_w2=float_w2,
        **scales,
    )
    np.savez_compressed(paths["splits"], **splits)
    convergence_warnings = [
        str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)
    ]
    metadata = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "seed": seed,
        "dataset": "sklearn.datasets.load_digits",
        "input_convention": "raw float64 pixel values; fixed /16 normalization folded into W1",
        "split_method": "stratified test_size=0.20, then calibration_size=0.125 of remainder",
        "split_sizes": {key.removesuffix("_indices"): len(value) for key, value in splits.items()},
        "hyperparameters": {**TRAINING_PARAMETERS, "random_state": seed},
        "training_threads": 1,
        "training_iterations": int(classifier.n_iter_),
        "training_loss": float(classifier.loss_),
        "converged": not convergence_warnings,
        "convergence_warnings": convergence_warnings,
        "scales": scales,
        "calibration_clipping_counts": {
            "input": clipping_count(x[calibration], input_scale),
            "hidden": clipping_count(cal_hidden, scales["hidden_scale"]),
        },
        "weight_clipping_counts": {
            "w1": clipping_count(float_w1, w1_scale),
            "w2": clipping_count(float_w2, w2_scale),
        },
        "calibration_hidden_path": "quantized layer1 exact accumulator, dequantize, bias, ReLU",
        "held_out_evaluated": False,
    }
    paths["training"].write_text(json.dumps(metadata, indent=2) + "\n")
    return paths
