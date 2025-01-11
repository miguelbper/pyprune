import os

import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest

from pyprune.backtracking import ArrayBitMask, ArrayInt, Backtracking, Int, rule


class Sudoku(Backtracking):
    def __init__(self, square_len: int) -> None:
        super().__init__()
        self.square_len = square_len

    @rule
    def sudoku(self, bm: ArrayBitMask) -> ArrayBitMask | None:
        """Applies the rules of Sudoku.

        If a cell (i, j) has value x, then
        - remove x from the other cells in the same row
        - remove x from the other cells in the same column
        - remove x from the other cells in the same box

        Args:
            bm (ArrayBitMask): The ArrayBitMask matrix representing the grid.

        Returns:
            ArrayBitMask | None: The updated ArrayBitMask matrix after applying
                the Sudoku rules.
        """
        k = self.square_len
        n = k**2

        for i in range(n):
            for j in range(n):
                c = bm[i, j]
                if not c:
                    return None
                if c & (c - 1) == 0:
                    mask = ~c
                    u = (i // k) * k
                    v = (j // k) * k
                    bm[i, :] &= mask
                    bm[:, j] &= mask
                    bm[u : u + k, v : v + k] &= mask
                    bm[i, j] = c
        return bm

    def is_sudoku(self, xm: ArrayInt) -> bool:
        def unique(arr: np.ndarray) -> np.ndarray:
            return np.apply_along_axis(lambda x: np.unique(x).size, axis=1, arr=arr)

        k = self.square_len
        n = k**2

        rows = xm
        cols = xm.T
        sqrs = np.array([xm[i * k : (i + 1) * k, j * k : (j + 1) * k].flatten() for i in range(k) for j in range(k)])
        catd = np.concatenate([rows, cols, sqrs], axis=0)
        return np.all(unique(catd) == n)


def parse_file_to_sudoku(filename: str) -> list[ArrayInt]:
    """Reads Sudoku puzzles from a file.

    The file should contain one Sudoku puzzle per line, with the 9*9
    digits in a row. In this file, 0 means that the cell is not filled.
    Example:

    sudoku.txt:
    000075400000000008080190000300001060000000034000068170204000603900000020530200000
    300000000050703008000028070700000043000000000003904105400300800100040000968000200
    ...

    Args:
        filename (str): The path to the file containing Sudoku puzzles.

    Returns:
        list[Grid]: A list of Sudoku grids.
    """
    sudokus = []
    with open(filename) as file:
        for line in file:
            line = line.strip()  # remove newline character
            if len(line) == 81:  # 9*9 digits
                sudoku = np.array(list(map(int, line))).reshape(9, 9)
                sudokus.append(sudoku.astype(Int))
    return sudokus


num_sudokus = 100
square_len = 3
num_digits = square_len**2
unknown = sum(1 << (d + 1) for d in range(num_digits)) * np.ones((num_digits, num_digits), dtype=Int)
file_sudokus = os.path.join(os.path.dirname(__file__), "sudoku.txt")
sudokus = parse_file_to_sudoku(file_sudokus)[:num_sudokus]


@pytest.fixture
def solver() -> Sudoku:
    return Sudoku(square_len=3)


@pytest.fixture(params=sudokus)
def sudoku(request: FixtureRequest) -> ArrayInt:
    return request.param


class TestSudoku:
    def test_sudoku(self, solver: Sudoku, sudoku: ArrayInt) -> None:
        xm = sudoku
        bm = np.where(xm, 1 << xm, unknown)
        sol = solver.solution([bm])
        sols = solver.solutions([bm])
        assert solver.is_sudoku(sol)
        assert len(sols) == 1
        assert np.array_equal(sol, sols[0])

    def test_progress(self, solver: Sudoku, sudoku: ArrayInt) -> None:
        xm = sudoku
        bm = np.where(xm, 1 << xm, unknown)
        _ = solver.solutions([bm], verbose=True)
        assert solver.num_pruned == solver.num_total
