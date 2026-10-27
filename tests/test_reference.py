import numpy as np
import pytest

from tinynpu.reference import matmul_int8


def test_signed_hand_calculation() -> None:
    a = np.array([[1, -2], [-3, 4]], dtype=np.int8)
    b = np.array([[5, -6], [-7, 8]], dtype=np.int8)
    actual = matmul_int8(a, b)
    np.testing.assert_array_equal(actual, [[19, -22], [-43, 50]])
    assert actual.dtype == np.int32


@pytest.mark.parametrize(("left", "right"), [(-128, -128), (-128, 127), (127, 127)])
def test_extreme_operands_do_not_overflow(left: int, right: int) -> None:
    a = np.full((4, 64), left, dtype=np.int8)
    b = np.full((64, 4), right, dtype=np.int8)
    np.testing.assert_array_equal(matmul_int8(a, b), np.full((4, 4), 64 * left * right))


@pytest.mark.parametrize("k", [1, 4, 16, 64])
def test_seeded_products_against_python_integer_sums(k: int) -> None:
    rng = np.random.default_rng(20261027 + k)
    for _ in range(25):
        a = rng.integers(-128, 128, size=(4, k), dtype=np.int8)
        b = rng.integers(-128, 128, size=(k, 4), dtype=np.int8)
        expected = [
            [sum(int(a[i, t]) * int(b[t, j]) for t in range(k)) for j in range(4)] for i in range(4)
        ]
        np.testing.assert_array_equal(matmul_int8(a, b), expected)


def test_zero_padding_preserves_useful_outputs() -> None:
    a = np.array([[3, -7]], dtype=np.int8)
    b = np.array([[1, 2, 3], [-1, -2, -3]], dtype=np.int8)
    padded = matmul_int8(np.pad(a, ((0, 3), (0, 0))), np.pad(b, ((0, 0), (0, 1))))
    np.testing.assert_array_equal(padded[:1, :3], [[10, 20, 30]])
    assert np.count_nonzero(padded[1:, :]) == 0
    assert np.count_nonzero(padded[:, 3:]) == 0


def test_implicit_narrowing_is_rejected() -> None:
    with pytest.raises(TypeError, match="dtype int8"):
        matmul_int8(np.array([[256]], dtype=np.int64), np.ones((1, 1), dtype=np.int8))


@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    [((4,), (4, 4)), ((4, 3), (4, 4)), ((4, 0), (0, 4)), ((4, 65), (65, 4)), ((0, 4), (4, 4))],
)
def test_invalid_dimensions_are_rejected(
    a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        matmul_int8(np.zeros(a_shape, dtype=np.int8), np.zeros(b_shape, dtype=np.int8))
