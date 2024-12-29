import numpy as np
import pytest

from pyprune.backtracking import Backtracking, Choices, Grid
from pyprune.subset import is_singleton, smallest


class Simple:
    @staticmethod
    def grid(cm: Choices) -> Grid:
        arr = np.array([[smallest(s) for s in row] for row in cm])
        return arr.astype(np.int32)

    @staticmethod
    def accept(cm: Choices) -> np.bool:
        bool_arr = np.array([[is_singleton(s) for s in row] for row in cm])
        return np.all(bool_arr)


class Problem(Backtracking):
    def __init__(self, *args, **kwargs) -> None:
        pass


@pytest.fixture(params=[1, 5], ids=lambda x: f"[n={x}]")
def n(request) -> int:
    return request.param


@pytest.fixture(params=[1, 2, 8], ids=lambda x: f"[k={x}]")
def k(request) -> int:
    return request.param


@pytest.fixture(params=list(range(5)), ids=lambda x: f"[seed={x}]")
def cm(request, n: int, k: int) -> Choices:
    rng = np.random.default_rng(request.param)
    return rng.integers(0, 2**k, (n, n), dtype=np.int32)


@pytest.fixture(params=list(range(5)), ids=lambda x: f"[seed={x}]")
def cm_singletons(request, n: int, k: int) -> Choices:
    rng = np.random.default_rng(request.param)
    return np.pow(2, rng.integers(0, k, (n, n), dtype=np.int32))


@pytest.fixture(params=list(range(5)), ids=lambda x: f"[seed={x}]")
def cm_nonzeros(request, n: int, k: int) -> Choices:
    rng = np.random.default_rng(request.param)
    return rng.integers(1, 2**k, (n, n), dtype=np.int32)


# @pytest.fixture()
# def problem() -> Problem:
#     return Problem()


class TestBacktracking:
    def test_grid(self, cm_singletons: Choices):
        assert np.array_equal(Backtracking.grid(cm_singletons), Simple.grid(cm_singletons))

    def test_accept(self, cm_nonzeros: Choices):
        assert Backtracking.accept(cm_nonzeros) == Simple.accept(cm_nonzeros)

    def test_reject(self):
        assert Backtracking.reject(None)
        assert Backtracking.reject(np.zeros((2, 2)))
        assert Backtracking.reject(np.eye(2))
        assert not Backtracking.reject(np.ones((2, 2)))

    def test_expand(self, cm_nonzeros: Choices):
        if np.all(cm_nonzeros & (cm_nonzeros - 1) == 0):
            return
        problem = Problem()
        cms = np.stack(problem.expand(cm_nonzeros), axis=0)  # type: ignore[arg-type]
        comparisons = cms == cm_nonzeros
        comparison = comparisons[0]
        assert np.all(comparisons == comparison)
        assert np.sum(~comparison) == 1
        i, j = np.where(~comparison)
        assert np.sum(cms[:, i, j]) == cm_nonzeros[i, j]
