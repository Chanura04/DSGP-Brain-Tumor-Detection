import numpy as np
from src.utils.batching import create_batch

import pytest


@pytest.mark.parametrize(
    ["iterable", "n", "expected"],
    [
        [np.arange(1, 5), 2, [[1, 2], [3, 4]]],
        [np.arange(1, 10), 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
        [np.arange(1, 10), 4, [[1, 2, 3, 4], [5, 6, 7, 8], [9]]],
    ],
)
def test_batch(iterable, n, expected):
    result = list(create_batch(iterable, n))

    assert isinstance(result, list)
    assert result == expected
