"""Host arithmetic and preparation isolation checks, without held-out evaluation."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_digits

import tinynpu.model as model_module
from tinynpu.model import (
    QuantizedModel,
    clipping_count,
    load_model,
    make_splits,
    prepare_model,
    quantize_symmetric,
    symmetric_scale,
)


def test_quantization_ties_extremes_and_clipping() -> None:
    values = np.array([-1000, -127.5, -126.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 126.5, 1000])
    expected = np.array([-127, -127, -126, -2, -2, 0, 0, 2, 2, 126, 127], dtype=np.int8)
    np.testing.assert_array_equal(quantize_symmetric(values, 1.0), expected)
    assert clipping_count(values, 1.0) == 3
    np.testing.assert_array_equal(
        quantize_symmetric(np.array([-0.75, -0.25, 0.25, 0.75]), 0.5), [-2, 0, 0, 2]
    )
    np.testing.assert_array_equal(
        quantize_symmetric(np.array([-1e300, 1e300]), 1e-100), [-127, 127]
    )


def test_zero_scale_calibration_is_safe() -> None:
    assert symmetric_scale(np.zeros((3, 4))) == 1.0
    assert symmetric_scale(np.array([-254.0, 127.0])) == 2.0
    np.testing.assert_array_equal(quantize_symmetric(np.zeros((3, 4)), 1.0), np.zeros((3, 4)))


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
def test_reject_invalid_scales(scale: float) -> None:
    with pytest.raises(ValueError):
        quantize_symmetric(np.array([1.0]), scale)


@pytest.mark.parametrize("values", [np.array([]), np.array([np.nan]), np.array([np.inf])])
def test_reject_invalid_calibration(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        symmetric_scale(values)


def test_split_is_reproducible_stratified_and_disjoint() -> None:
    _, labels = load_digits(return_X_y=True)
    first = make_splits(labels)
    second = make_splits(labels)
    assert {key: len(value) for key, value in first.items()} == {
        "train_indices": 1257,
        "calibration_indices": 180,
        "test_indices": 360,
    }
    for key, indices in first.items():
        np.testing.assert_array_equal(indices, second[key])
        proportions = np.bincount(labels[indices], minlength=10) / len(indices)
        np.testing.assert_allclose(proportions, np.bincount(labels) / len(labels), atol=0.01)
    all_indices = np.concatenate(list(first.values()))
    np.testing.assert_array_equal(np.sort(all_indices), np.arange(len(labels)))


@pytest.fixture
def small_model() -> QuantizedModel:
    w1 = np.zeros((64, 16), dtype=np.int8)
    w2 = np.zeros((16, 10), dtype=np.int8)
    w1[0, 0] = 2
    w2[0, 1] = 2
    b1 = np.zeros(16, dtype=np.float64)
    b1[0] = -1
    b2 = np.zeros(10, dtype=np.float64)
    b2[2] = 1
    return QuantizedModel(w1, w2, b1, b2, 1.0, 1.0, 1.0, 1.0, w1.astype(float), w2.astype(float))


def test_shared_host_pipeline_and_intermediate_values(small_model: QuantizedModel) -> None:
    x = np.zeros((2, 64))
    x[:, 0] = [2, -2]
    result = small_model.infer_software(x)
    expected_acc1 = np.zeros((2, 16), dtype=np.int32)
    expected_acc1[:, 0] = [4, -4]
    np.testing.assert_array_equal(result.layer1_acc, expected_acc1)
    expected_hidden = np.zeros((2, 16), dtype=np.int8)
    expected_hidden[0, 0] = 3
    np.testing.assert_array_equal(result.hidden_quantized, expected_hidden)
    expected_acc2 = np.zeros((2, 10), dtype=np.int32)
    expected_acc2[0, 1] = 6
    np.testing.assert_array_equal(result.layer2_acc, expected_acc2)
    expected_logits = expected_acc2.astype(float)
    expected_logits[:, 2] = 1
    np.testing.assert_array_equal(result.logits, expected_logits)
    np.testing.assert_array_equal(result.predictions, [1, 2])
    np.testing.assert_array_equal(small_model.predict_float(x), [1, 2])
    np.testing.assert_array_equal(small_model.hidden_from_acc(result.layer1_acc), expected_hidden)
    np.testing.assert_array_equal(small_model.logits_from_acc(result.layer2_acc), expected_logits)


@pytest.mark.parametrize("shape", [(64,), (1, 63), (0, 64)])
def test_input_shape_validation(small_model: QuantizedModel, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        small_model.quantize_input(np.zeros(shape))


def test_accumulator_dtype_and_shape_validation(small_model: QuantizedModel) -> None:
    with pytest.raises(TypeError):
        small_model.hidden_from_acc(np.zeros((1, 16), dtype=np.int8))
    with pytest.raises(ValueError):
        small_model.hidden_from_acc(np.zeros((1, 15), dtype=np.int32))
    with pytest.raises(TypeError):
        small_model.logits_from_acc(np.zeros((1, 10), dtype=np.float64))
    with pytest.raises(ValueError):
        small_model.logits_from_acc(np.zeros((1, 9), dtype=np.int32))


def test_preparation_train_and_calibration_isolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Deliberately corrupt test pixels. Preparation must not touch their values.
    x, y = load_digits(return_X_y=True)
    splits = make_splits(y)
    x[splits["test_indices"]] = np.nan
    fitted: list[np.ndarray] = []

    class FakeClassifier:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["random_state"] == 20261027
            self.coefs_ = [np.full((64, 16), 0.25), np.full((16, 10), 0.5)]
            self.intercepts_ = [np.zeros(16), np.zeros(10)]
            self.n_iter_ = 1
            self.loss_ = 0.5

        def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
            np.testing.assert_array_equal(train_x, x[splits["train_indices"]] / 16.0)
            np.testing.assert_array_equal(train_y, y[splits["train_indices"]])
            fitted.append(train_x)

    monkeypatch.setattr(model_module, "load_digits", lambda **kwargs: (x, y))
    monkeypatch.setattr(model_module, "MLPClassifier", FakeClassifier)
    paths = prepare_model(tmp_path)
    assert len(fitted) == 1
    restored = load_model(paths["model"])
    assert restored.input_scale == 16.0 / 127.0
    assert restored.hidden_scale > 0
    np.testing.assert_array_equal(restored.float_w1, np.full((64, 16), 0.25 / 16.0))
    metadata = json.loads(paths["training"].read_text())
    assert metadata["held_out_evaluated"] is False
    assert metadata["converged"] is True
    assert metadata["split_sizes"] == {"train": 1257, "calibration": 180, "test": 360}
    assert "accuracy" not in metadata
    with np.load(paths["splits"], allow_pickle=False) as saved:
        for key, indices in splits.items():
            np.testing.assert_array_equal(saved[key], indices)
    # Inference accepts a reloaded numeric-only model with matching intermediates.
    software = restored.infer_software(x[splits["calibration_indices"][:4]])
    assert software.layer1_acc.shape == (4, 16)
    assert software.layer2_acc.shape == (4, 10)
