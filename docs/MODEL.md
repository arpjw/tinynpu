# Software classifier and quantization

`src/tinynpu/model.py` provides the fixed 64 → 16 → 10 ReLU classifier.
Preparation exports numeric-only `model.npz`, exact index arrays in `splits.npz`,
and training/calibration metadata in `training.json`. It does not evaluate the
held-out split or make an accuracy claim.

## Frozen preparation choices

- Dataset: scikit-learn `load_digits`, raw float64 pixels in the original 0..16 scale.
- Seed: 20261027. First stratify off 20% test; then stratify off 12.5% of the
  remaining examples for calibration. This yields 1,257 train, 180 calibration,
  and 360 test examples. Preserve returned index order.
- Training: `MLPClassifier` with one 16-unit ReLU layer, L-BFGS, alpha 0.0001,
  max_iter 500, max_fun 15000, tol 0.0001, and one BLAS thread.
- Divide training pixels by the fixed constant 16, then fold that normalization
  into exported first-layer weights. All inference APIs accept raw pixels.
- Quantization: per-tensor symmetric absmax scales, ties-to-even rounding, and
  clipping to [-127, 127]. An all-zero tensor uses scale 1.
- Fit input scale on calibration pixels. Fit hidden scale on calibration data
  passed through the quantized first-layer product, dequantization, bias, and
  ReLU. Weight scales use trained weights only.
- Preserve original floating-point biases. The second-layer product is
  dequantized and biased on the host; predictions use its argmax.

`training.json` records convergence warnings, training iterations, scales, and
calibration clipping counts. Training is a single fixed run with no accuracy
selection or test-driven retry. Absence of a convergence warning means the solver
reported convergence; it does not establish global optimality.

## Integration API

```python
from pathlib import Path
from tinynpu.model import load_model, prepare_model

paths = prepare_model(Path("artifacts/model"))
model = load_model(paths["model"])
# x: nonempty (N, 64) array of raw digit pixels
software = model.infer_software(x)
q_input = model.quantize_input(x)
# Send q_input @ model.w1 to RTL; concatenate its output tiles into acc1.
q_hidden = model.hidden_from_acc(acc1)
# Send q_hidden @ model.w2 to RTL; concatenate its output tiles into acc2.
logits = model.logits_from_acc(acc2)
```

`SoftwareResult` retains logits, predictions, both INT32 accumulator arrays,
quantized input, and quantized hidden activations. Compare both accumulator
arrays exactly against hardware before comparing predictions. Use
`model.predict_float(x)` for the floating-point baseline.

Preparation tests use a fake estimator and deliberately replace test pixels
with NaN. Successful preparation and finite exports demonstrate that test pixel
values do not enter training or calibration. Separate tests cover split coverage,
disjointness, reproducibility, quantization ties/extremes, host operations,
intermediate results, and malformed inputs.
