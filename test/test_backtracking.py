from timeit import timeit

import numpy as np
import pytest

from pyprune.backtracking import Backtracking, Choices, Grid
from pyprune.subset import is_singleton, smallest


class Simple:
    @staticmethod
    def grid_(cm: Choices) -> Grid:
        arr = np.array([[smallest(s) for s in row] for row in cm])
        return arr.astype(np.uint32)

    @staticmethod
    def accept_(cm: Choices) -> np.bool:
        bool_arr = np.array([[is_singleton(s) for s in row] for row in cm])
        return np.all(bool_arr)


@pytest.fixture(params=[1, 5], ids=lambda x: f"[n={x}]")
def n(request) -> int:
    return request.param


@pytest.fixture(params=[1, 2, 8], ids=lambda x: f"[k={x}]")
def k(request) -> int:
    return request.param


@pytest.fixture(params=list(range(5)), ids=lambda x: f"[seed={x}]")
def cm(request, n: int, k: int) -> Choices:
    rng = np.random.default_rng(request.param)
    return rng.integers(0, 2**k, (n, n), dtype=np.uint32)


class TestBacktracking:
    def test_accept(self, cm: Choices):
        if np.all(cm):
            assert Backtracking.accept(cm) == Simple.accept_(cm)

    def test_expand(self, cm: Choices):
        if np.all(cm) and not np.all(np.logical_and(cm, cm & (cm - 1) == 0)):
            cms = np.stack(Backtracking.expand(cm), axis=0)
            comparisons = cms == cm
            comparison = comparisons[0]
            assert np.all(comparisons == comparison)
            assert np.sum(~comparison) == 1
            multi_index = np.where(~comparison)
            assert np.sum(cms[:, *multi_index]) == cm[multi_index]


numba_functions = [
    Backtracking.grid,
    Backtracking.accept,
]


@pytest.fixture(params=numba_functions)
def func(request):
    return request.param


class TestBacktrackingNumba:
    def test_numba(self, func):
        # arrange
        n_iters = 10000
        rng = np.random.default_rng(1337)
        cm = rng.integers(0, 2**16, (10, 10), dtype=np.uint32)
        func(cm)
        slow = func.py_func
        # act
        t_fast = timeit(lambda: func(cm), number=n_iters)
        t_slow = timeit(lambda: slow(cm), number=n_iters)
        # assert
        assert t_fast < t_slow
