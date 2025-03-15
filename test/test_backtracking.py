import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest

from pyprune.backtracking import ArrayBitMask, ArrayInt, Backtracking, Int

grid = Backtracking.grid
accept = Backtracking.accept
reject = Backtracking.reject


n, m = 2, 2
num_random_arrays = 10


@pytest.fixture(params=[2, 8], ids=lambda x: f"[k={x}]")
def num_bits(request: FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=list(range(num_random_arrays)), ids=lambda x: f"[seed={x}]")
def xm(request: FixtureRequest, num_bits: int) -> ArrayInt:
    rng = np.random.default_rng(request.param)
    return rng.integers(0, num_bits, (n, m), dtype=Int)


@pytest.fixture
def bm(xm: ArrayInt) -> ArrayBitMask:
    return 1 << xm


@pytest.fixture
def cm(bm: ArrayBitMask) -> ArrayBitMask:
    bm_copy = np.copy(bm)
    bm_copy[0, 0] = 3
    return bm_copy


class TestGrid:
    def test_inverse(self, xm: ArrayInt, bm: ArrayBitMask) -> None:
        assert np.array_equal(grid(bm), xm)


class TestAccept:
    def test_accept_powers(self, bm: ArrayBitMask) -> None:
        assert accept(bm)

    def test_dont_accept_inverse_power(self, bm: ArrayBitMask) -> None:
        assert not accept(np.invert(bm))

    def test_dont_accept_branchable(self, cm: ArrayBitMask) -> None:
        assert not accept(cm)


class TestReject:
    def test_reject_none(self) -> None:
        assert reject(None)

    def test_reject_zeros(self) -> None:
        assert reject(np.zeros((n, m), dtype=Int))

    def test_dont_reject_powers(self, bm: ArrayBitMask) -> None:
        assert not reject(bm)

    def test_dont_reject_branchable(self, cm: ArrayBitMask) -> None:
        assert not reject(cm)


class TestBranch:
    def test_branch(self, cm: ArrayBitMask) -> None:
        pb = Backtracking()
        cms = np.stack(pb.branch(cm), axis=0)
        comparisons = cms == cm
        comparison = comparisons[0]
        assert np.all(comparisons == comparison)
        assert np.sum(~comparison) == 1
        i, j = np.where(~comparison)
        assert np.sum(cms[:, i, j]) == cm[i, j]
