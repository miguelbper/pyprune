import time
from pathlib import Path

from pyprune.sudoku.sudoku import sudoku_solver
from pyprune.sudoku.utils import is_sudoku, parse_file_to_sudoku

NUM_SUDOKUS = 1000


def benchmark(file_sudokus: Path | str, num_sudokus: int) -> float:
    sudokus = parse_file_to_sudoku(file_sudokus)[:num_sudokus]
    time_start = time.time()
    for sudoku in sudokus:
        solution = sudoku_solver(sudoku)
        assert is_sudoku(solution)
    time_end = time.time()
    return time_end - time_start


def main() -> None:
    file_sudokus = Path(__file__).parent / "sudoku.txt"
    num_sudokus = 1000
    time_taken = benchmark(file_sudokus, num_sudokus)
    print(f"Time taken: {time_taken} seconds")
    # Time taken: 1.8059041500091553 seconds


if __name__ == "__main__":
    main()
