# Third-party notices

TinyNPU's original code and documentation are covered by the [MIT license](LICENSE).
The dataset and external dependencies retain their own licenses.

## Handwritten digit data

The experiment uses the 1,797-example digits dataset distributed with
[scikit-learn](https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset),
derived from the UCI Optical Recognition of Handwritten Digits dataset:

E. Alpaydin and C. Kaynak (1998), *Optical Recognition of Handwritten Digits*,
UCI Machine Learning Repository. [DOI: 10.24432/C50P49](https://doi.org/10.24432/C50P49).

The [UCI dataset page](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
identifies its license as [Creative Commons Attribution 4.0 International
(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
This attribution and license apply to the digit pixels and labels in generated
experiment artifacts and the hosted demo. The MIT license does not replace the
dataset license.

TinyNPU splits the scikit-learn data for training, calibration, and evaluation.
The demo selects examples and renders their original 8×8 pixel values at a larger
display size. Model predictions, quantized operands, and RTL traces are added by
TinyNPU. The dataset creators do not endorse this project.

## Software dependencies

Python dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.
Verilator and the C++ compiler are separately installed build tools. Consult
their distributions for their license texts and notices. TinyNPU's static viewer
uses browser APIs and does not bundle a third-party JavaScript framework.
