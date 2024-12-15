import os
from itertools import product

import numpy as np
import pytest

from pyprune.backtracking import Backtracking, Choices
from pyprune.examples.sudoku import Sudoku, is_sudoku_solution, parse_file_to_sudoku
from pyprune.subset import num_elements, remove_except


class NothingIsSolution(Backtracking):
    def __init__(self, cm: Choices) -> None:
        super().__init__(cm)

    @staticmethod
    def prune(cm: Choices) -> Choices | None:
        return None


class EverythingIsSolution(Backtracking):
    def __init__(self, cm: Choices) -> None:
        super().__init__(cm)

    @staticmethod
    def prune(cm: Choices) -> Choices | None:
        return cm


class OnlyZeros(Backtracking):
    def __init__(self, cm: Choices) -> None:
        super().__init__(cm)

    @staticmethod
    def prune(cm: Choices) -> Choices | None:
        m, n = cm.shape
        ans = np.copy(cm)
        for i, j in product(range(m), range(n)):
            ans[i, j] = remove_except(cm[i, j], 0)
        return ans


@pytest.fixture(params=[1, 2, 5], ids=lambda x: f"[n={x}]")
def n(request) -> int:
    return request.param


@pytest.fixture(params=[2, 8], ids=lambda x: f"[k={x}]")
def k(request) -> int:
    return request.param


@pytest.fixture(params=list(range(5)), ids=lambda x: f"[seed={x}]")
def cm(request, n: int, k: int) -> Choices:
    rng = np.random.default_rng(request.param)
    return rng.integers(1, 2**k, (n, n), dtype=np.int32)


class TestIntegration:
    def test_nothing_is_solution(self, cm: Choices):
        problem = NothingIsSolution(cm)
        assert problem.solution() is None
        assert problem.solutions() == []

    def test_everything_is_solution(self):
        rng = np.random.default_rng(1337)
        cm = rng.integers(1, 2**2, (2, 2), dtype=np.int32)
        problem = EverythingIsSolution(cm)
        num_choices = np.vectorize(num_elements)(cm)
        assert len(problem.solutions()) == np.prod(num_choices)

    def test_only_zeros(self, cm: Choices):
        zeros = np.zeros(cm.shape, dtype=np.int32)
        ones = np.ones(cm.shape, dtype=np.int32)
        cm += np.where(cm % 2, zeros, ones)
        problem = OnlyZeros(cm)
        soln = problem.solution()
        assert soln is not None
        assert np.array_equal(soln, zeros)
        assert len(problem.solutions()) == 1

    def test_sudoku(self):
        sudokus = parse_file_to_sudoku(os.path.join("pyprune", "examples", "sudoku.txt"))
        for sudoku in sudokus[:100]:  # file has 10000 sudokus, take only 100
            cm = np.where(sudoku, 2**sudoku, (2**10 - 2) * np.ones((9, 9)))
            problem = Sudoku(cm)
            solution = problem.solution()
            assert is_sudoku_solution(solution)
            assert len(problem.solutions()) == 1
