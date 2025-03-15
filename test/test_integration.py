import numpy as np

from pyprune.backtracking import ArrayBitMask, Backtracking, Int


class NothingIsSolution(Backtracking):
    def prune(self, bm: ArrayBitMask) -> ArrayBitMask | None:
        return None


class EverythingIsSolution(Backtracking):
    def prune(self, bm: ArrayBitMask) -> ArrayBitMask | None:
        return bm


class OnlyZeros(Backtracking):
    def prune(self, bm: ArrayBitMask) -> ArrayBitMask | None:
        return bm & (1 << 0)


nothing = NothingIsSolution()
everything = EverythingIsSolution()
only_zeros = OnlyZeros()


n, m = 2, 2
num_bits = 2
unknown = sum(1 << d for d in range(num_bits)) * np.ones((n, m), dtype=Int)
bm_zero = (1 << 0) * np.ones((n, m), dtype=Int)
bm_ones = (1 << 1) * np.ones((n, m), dtype=Int)


class TestNothingIsSolution:
    def test_unknown(self) -> None:
        sol = nothing.solution([unknown])
        sols = nothing.solutions([unknown])
        assert sol is None
        assert sols == []

    def test_bm_zero(self) -> None:
        sol = nothing.solution([bm_zero])
        sols = nothing.solutions([bm_zero])
        assert sol is None
        assert sols == []

    def test_bm_ones(self) -> None:
        sol = nothing.solution([bm_ones])
        sols = nothing.solutions([bm_ones])
        assert sol is None
        assert sols == []


class TestEverythingIsSolution:
    def test_unknown(self) -> None:
        sols = everything.solutions([unknown])
        assert len(sols) == num_bits ** (n * m)

    def test_bm_zero(self) -> None:
        sol = everything.solution([bm_zero])
        sols = everything.solutions([bm_zero])
        assert sol is not None
        assert np.array_equal(1 << sol, bm_zero)
        assert len(sols) == 1

    def test_bm_ones(self) -> None:
        sol = everything.solution([bm_ones])
        sols = everything.solutions([bm_ones])
        assert sol is not None
        assert np.array_equal(1 << sol, bm_ones)
        assert len(sols) == 1


class TestOnlyZeros:
    def test_unknown(self) -> None:
        sol = only_zeros.solution([unknown])
        sols = only_zeros.solutions([unknown])
        assert sol is not None
        assert np.array_equal(1 << sol, bm_zero)
        assert len(sols) == 1

    def test_bm_zero(self) -> None:
        sol = only_zeros.solution([bm_zero])
        sols = only_zeros.solutions([bm_zero])
        assert sol is not None
        assert np.array_equal(1 << sol, bm_zero)
        assert len(sols) == 1

    def test_bm_ones(self) -> None:
        sol = only_zeros.solution([bm_ones])
        sols = only_zeros.solutions([bm_ones])
        assert sol is None
        assert sols == []
