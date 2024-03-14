import os
from typing import Optional
from itertools import product
import numpy as np
from numba import njit
from backtracking.subset import (
    remove_except,
    num_elements,
    is_singleton_numba,
    smallest_numba
)
from backtracking.backtracking import Backtracking, Choices


rng = np.random.default_rng(1337)


class NothingIsSolution(Backtracking):
    def __init__(self, cm0: Choices) -> None:
        super().__init__(cm0)

    def constraint_nothing_is_solution(self, cm: Choices) -> Optional[Choices]:
        return None


class EverythingIsSolution(Backtracking):
    def __init__(self, cm0: Choices) -> None:
        super().__init__(cm0)


class OnlyZeros(Backtracking):
    def __init__(self, cm0: Choices) -> None:
        super().__init__(cm0)

    def constraint_only_zeros(self, cm: Choices) -> Optional[Choices]:
        m, n = cm.shape
        ans = np.copy(cm)
        for i, j in product(range(m), range(n)):
            ans[i, j] = remove_except(cm[i, j], 0)
        return ans


class Sudoku(Backtracking):
    def __init__(self, cm0: Choices) -> None:
        super().__init__(cm0)

    @staticmethod
    @njit
    def constraint_sudoku(cm: Choices) -> Optional[Choices]:
        cm = np.copy(cm)
        for i in range(9):
            for j in range(9):
                if is_singleton_numba(cm[i, j]):
                    x = smallest_numba(cm[i, j])

                    # remove x from row
                    for k in range(9):
                        if k != j:
                            cm[i, k] &= ~(1 << x)

                    # remove x from column
                    for k in range(9):
                        if k != i:
                            cm[k, j] &= ~(1 << x)

                    # remove x from box
                    box_i, box_j = 3 * (i // 3), 3 * (j // 3)
                    for di in range(3):
                        for dj in range(3):
                            i_ = box_i + di
                            j_ = box_j + dj
                            if not (i_ == i and j_ == j):
                                cm[i_, j_] &= ~(1 << x)
        return cm


def parse_file_to_sudoku(filename):
    sudokus = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # remove newline character
            if len(line) == 81:  # 9*9 digits
                sudoku = np.array(list(map(int, line))).reshape(9, 9)
                sudokus.append(sudoku)
    return sudokus


def is_sudoku_solution(cm: np.ndarray) -> bool:
    # Check rows
    for i in range(9):
        if len(np.unique(cm[i])) != 9:
            return False

    # Check columns
    for j in range(9):
        if len(np.unique(cm[:, j])) != 9:
            return False

    # Check boxes
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            box = cm[i:i+3, j:j+3]
            if len(np.unique(box)) != 9:
                return False

    return True


class TestIntegration:
    def test_nothing_is_solution(self):
        for _ in range(10):
            cm = rng.integers(1, 2**4, (5, 5), dtype=np.uint32)
            problem = NothingIsSolution(cm)
            assert problem.solution() is None
            assert problem.solutions() == []

    def test_everything_is_solution(self):
        for _ in range(5):
            cm = rng.integers(1, 2**3, (2, 2), dtype=np.uint32)
            problem = EverythingIsSolution(cm)
            num_choices = np.vectorize(num_elements)(cm)
            assert len(problem.solutions()) == np.prod(num_choices)

    def test_only_zeros(self):
        for _ in range(10):
            cm = rng.integers(1, 2**4, (5, 5), dtype=np.uint32)
            zeros = np.zeros((5, 5), dtype=np.uint32)
            ones = np.ones((5, 5), dtype=np.uint32)
            cm += np.where(cm % 2, zeros, ones)
            problem = OnlyZeros(cm)
            assert np.array_equal(problem.solution(), zeros)
            assert len(problem.solutions()) == 1

    def test_sudoku(self):
        sudokus = parse_file_to_sudoku(os.path.join('test', 'sudoku.txt'))
        for sudoku in sudokus[:100]:  # file has 10000 sudokus, take only 100
            cm0 = np.where(sudoku, 2**sudoku, (2**10 - 2)*np.ones((9, 9)))
            problem = Sudoku(cm0)
            solution = problem.solution()
            assert is_sudoku_solution(solution)
            assert len(problem.solutions()) == 1
