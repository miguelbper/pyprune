import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest
from test_sudoku import Sudoku

from pyprune.backtracking import ArrayBitMask, ArrayInt, Backtracking, Int


class SudokuOptimizeProd(Sudoku):
    def __init__(self, square_len: int, maximize: bool) -> None:
        super().__init__(square_len)
        self.maximize = maximize

    def criterion(self, bm: ArrayBitMask, best_score: Int | float) -> Int | None:
        # criterion is prod of the numbers formed by the digits in rows
        # maximize => criterion(bm) >= criterion(xm) for all xm => replace unknown by maximal value
        # minimize => criterion(bm) <= criterion(xm) for all xm => replace unknown by minimal value
        n, _ = bm.shape
        replace = n if self.maximize else 1
        known = bm & (bm - 1) == 0
        xm = np.where(known, np.log2(bm), replace).astype(Int)
        pows = 10 ** np.arange(n - 1, -1, -1, dtype=Int)
        nums = xm @ pows
        return np.prod(nums)

    @staticmethod
    def score(xm: ArrayInt) -> np.int64:
        n, _ = xm.shape
        pows = 10 ** np.arange(n - 1, -1, -1, dtype=Int)
        nums = xm @ pows
        return np.prod(nums)


square_len = 2
num_digits = square_len**2
unknown = sum(1 << (d + 1) for d in range(num_digits)) * np.ones((num_digits, num_digits), dtype=Int)


@pytest.fixture(params=[False, True])
def maximize(request: FixtureRequest) -> bool:
    return request.param


@pytest.fixture
def problem(maximize: bool) -> SudokuOptimizeProd:
    return SudokuOptimizeProd(square_len, maximize)


class TestSudokuOptimizeProd:
    def test_optimize_exhaustive(self, maximize: bool, problem: SudokuOptimizeProd) -> None:
        _, best_score = problem.optimize([unknown], maximize)

        sols = problem.solutions([unknown])
        best = max if maximize else min
        best_score_exhaustive = best(map(problem.score, sols))

        assert best_score == best_score_exhaustive


class SumOfSquares(Backtracking):
    def __init__(self) -> None:
        super().__init__()
        nums = np.arange(100000)
        digs = np.arange(1, 11).reshape(-1, 1)
        self.divisible = nums % digs == 0  # (10, 100000)
        nums_digits = np.array([list(f"{i:05}") for i in range(100000)], dtype=Int)  # (100000, 5)
        self.pows_nums_digits = 1 << nums_digits  # (100000, 5)

    def prune(self, bm: ArrayBitMask) -> ArrayBitMask | None:
        catd = np.concatenate([bm, bm.T], axis=0)  # (10, 5)
        compatible = np.all(self.pows_nums_digits & catd.reshape(10, 1, 5), axis=2)  # (10, 100000)
        comp_divis = compatible & self.divisible  # (10, 100000)
        new_catd = np.bitwise_or.reduce(comp_divis.reshape(10, -1, 1) * self.pows_nums_digits, axis=1)  # (10, 5)
        top = new_catd[:5]  # (5, 5)
        bot = new_catd[5:].T  # (5, 5)
        bm = np.bitwise_and(top, bot)  # (5, 5)
        return bm

    @staticmethod
    def criterion(bm: ArrayBitMask, best_score: Int | float) -> Int | None:
        # criterion is prod of the numbers formed by the digits in rows
        # maximize => criterion(bm) >= criterion(xm) for all xm => replace unknown by maximal value
        xm = np.vectorize(lambda b: int(b).bit_length())(bm) - 1
        return np.sum(xm)


class TestSumOfSquares:
    def test_optimize(self) -> None:
        pb = SumOfSquares()

        bm0 = sum(1 << d for d in range(10)) * np.ones((5, 5), dtype=Int)
        bm0[4, 4] = 1 << 0
        bm0[1, 4] = sum(1 << d for d in range(0, 10, 2))
        bm0[3, 4] = sum(1 << d for d in range(0, 10, 2))
        bm0[4, 0] = sum(1 << d for d in range(0, 10, 2))
        bm0[4, 2] = sum(1 << d for d in range(0, 10, 2))

        # help the algo
        bm0[0, 1] = 1 << 8
        bm0[2, 0] = 1 << 7
        bm0[3, 4] = 1 << 6

        _, best_score = pb.optimize([bm0], maximize=True, verbose=True)
        assert best_score == 205
