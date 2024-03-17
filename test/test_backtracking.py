import numpy as np
from timeit import timeit
import pytest
from backtracking.subset import (
    smallest,
    is_singleton,
    num_elements,
    subset,
    elements,
)
from backtracking.backtracking import (
    Backtracking,
    Choices,
    Grid,
    argmin_num_elements,
)


class Simple:
    @staticmethod
    def grid_(cm: Choices) -> Grid:
        arr = np.array([[smallest(s) for s in row] for row in cm])
        return arr.astype(np.uint32)

    @staticmethod
    def accept_(cm: Choices) -> bool:
        bool_arr = np.array([[is_singleton(s) for s in row] for row in cm])
        return np.all(bool_arr)


@pytest.fixture(params=[1, 5], ids=lambda x: f"[n={x}]")
def n(request):
    return request.param


@pytest.fixture(params=[1, 2, 8], ids=lambda x: f"[k={x}]")
def k(request):
    return request.param


@pytest.fixture(params=list(range(5)), ids=lambda x: f"[seed={x}]")
def cm(request, n, k):
    rng = np.random.default_rng(request.param)
    return rng.integers(0, 2**k, (n, n), dtype=np.uint32)


class TestBacktracking:
    def test_grid(self, cm):
        assert np.array_equal(Backtracking.grid(cm), Simple.grid_(cm))

    def test_accept(self, cm):
        assert Backtracking.accept(cm) == Simple.accept_(cm)

    def test_argmin_num_elements(self, cm):
        nm = np.vectorize(num_elements)(cm)
        if np.all(nm < 2):
            return
        xm = np.where(nm < 2, np.inf, nm)
        i, j = argmin_num_elements(cm)
        assert num_elements(cm[i, j]) == np.min(xm)

    def test_expand(self, cm):
        cms = Backtracking.expand(cm)
        i, j = argmin_num_elements(cm)
        assert len(cms) == num_elements(cm[i, j])
        for cm, x in zip(cms, elements(cm[i, j])):
            assert cm[i, j] == subset([x])


numba_functions = [
    Backtracking.expand,
    Backtracking.accept,
    argmin_num_elements,
]


@pytest.fixture(params=numba_functions)
def func(request):
    return request.param


class TestBacktrackingNumba:
    def test_numba(self, func):
        # arrange
        n_iters = 10000
        rng = np.random.default_rng(1337)
        cm = rng.integers(0, 2 ** 16, (10, 10), dtype=np.uint32)
        func(cm)
        slow = func.py_func
        # act
        t_fast = timeit(lambda: func(cm), number=n_iters)
        t_slow = timeit(lambda: slow(cm), number=n_iters)
        # assert
        assert t_fast < t_slow
