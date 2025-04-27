from pathlib import Path

import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest

from pyprune.backtracking import ArrayInt, Int
from pyprune.sudoku.benchmark import benchmark
from pyprune.sudoku.sudoku import Sudoku
from pyprune.sudoku.utils import is_sudoku, parse_file_to_sudoku

NUM_SUDOKUS = 100
unknown = sum(1 << (d + 1) for d in range(9)) * np.ones((9, 9), dtype=Int)
file_sudokus = Path(__file__).parent.parent / "src" / "pyprune" / "sudoku" / "sudoku.txt"
sudokus = parse_file_to_sudoku(file_sudokus)[:NUM_SUDOKUS]


@pytest.fixture
def solver() -> Sudoku:
    return Sudoku()


@pytest.fixture(params=sudokus)
def sudoku(request: FixtureRequest) -> ArrayInt:
    return request.param


class TestSudoku:
    def test_sudoku(self, solver: Sudoku, sudoku: ArrayInt) -> None:
        xm = sudoku
        bm = np.where(xm, 1 << xm, unknown)
        sol = solver.solution([bm])
        sols = solver.solutions([bm])
        assert sol is not None
        assert is_sudoku(sol)
        assert len(sols) == 1
        assert np.array_equal(sol, sols[0])

    def test_none(self) -> None:
        assert not is_sudoku(None)


class TestSudokuBenchmark:
    def test_benchmark(self) -> None:
        benchmark(file_sudokus, NUM_SUDOKUS)


class TestRunFiles:
    def test_run_sudoku(self) -> None:
        from pyprune.sudoku.sudoku import main

        main()

    def test_run_benchmark(self) -> None:
        from pyprune.sudoku.benchmark import main

        main()
