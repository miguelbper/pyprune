import numpy as np
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
from timeit import timeit


rng = np.random.default_rng(1337)


class Simple:
    @staticmethod
    def grid_(cm: Choices) -> Grid:
        return np.array([[smallest(s) for s in row] for row in cm])

    @staticmethod
    def accept_(cm: Choices) -> bool:
        bool_arr = np.array([[is_singleton(s) for s in row] for row in cm])
        return np.all(bool_arr)


class TestBacktracking:
    def test_grid(self):
        for _ in range(10):
            cm = rng.integers(0, 2 ** 32, (5, 5), dtype=np.uint32)
            assert np.array_equal(Backtracking.grid(cm), Simple.grid_(cm))

    def test_accept(self):
        for _ in range(10):
            cm = rng.integers(0, 2 ** 32, (5, 5), dtype=np.uint32)
            assert Backtracking.accept(cm) == Simple.accept_(cm)

    def test_argmin_num_elements(self):
        for _ in range(10):
            cm = rng.integers(0, 2 ** 8, (5, 5), dtype=np.uint32)
            nm = np.vectorize(num_elements)(cm)
            nm = np.where(nm < 2, np.inf, nm)
            i, j = argmin_num_elements(cm)
            assert num_elements(cm[i, j]) == np.min(nm)

    def test_expand(self):
        for _ in range(10):
            cm0 = rng.integers(0, 2 ** 8, (5, 5), dtype=np.uint32)
            cms = Backtracking.expand(cm0)
            i, j = argmin_num_elements(cm0)
            assert len(cms) == num_elements(cm0[i, j])
            for cm, x in zip(cms, elements(cm0[i, j])):
                assert cm[i, j] == subset([x])


class TestBacktrackingNumba:
    def test_numba(self):
        funcs = [Backtracking.expand, Backtracking.accept, argmin_num_elements]
        for fast_fn in funcs:
            slow_fn = fast_fn.py_func
            fast_fn(np.zeros((1, 1), dtype=np.uint32))
            cm = rng.integers(0, 2 ** 16, (10, 10), dtype=np.uint32)
            n = 10000
            t_fast = timeit(lambda: fast_fn(cm), number=n)
            t_slow = timeit(lambda: slow_fn(cm), number=n)
            assert t_fast < t_slow
