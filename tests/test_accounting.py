import pytest

from tinynpu.inference import transfer_cost


def test_first_layer_single_image_hand_calculation() -> None:
    assert transfer_cost(1, 64, 4, 4) == {
        "load_bytes": 320,
        "store_bytes": 16,
        "load_cycles": 80,
        "store_cycles": 4,
    }


def test_padding_is_not_transferred_and_rounding_is_per_phase() -> None:
    assert transfer_cost(1, 1, 1, 8) == {
        "load_bytes": 2,
        "store_bytes": 4,
        "load_cycles": 1,
        "store_cycles": 1,
    }


@pytest.mark.parametrize("bandwidth", [0, -1, 1.5, True])
def test_invalid_bandwidth_is_rejected(bandwidth: int) -> None:
    with pytest.raises(ValueError):
        transfer_cost(1, 16, 2, bandwidth)


def test_fixed_network_byte_budget() -> None:
    for rows, expected in [(1, (1488, 104)), (4, (2400, 416))]:
        costs = [transfer_cost(rows, 64, 4, 1) for _ in range(4)]
        costs += [transfer_cost(rows, 16, n, 1) for n in (4, 4, 2)]
        assert sum(c["load_bytes"] for c in costs) == expected[0]
        assert sum(c["store_bytes"] for c in costs) == expected[1]
