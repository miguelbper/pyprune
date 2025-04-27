from pathlib import Path

import numpy as np

from pyprune.backtracking import ArrayInt, Int


def is_sudoku(xm: ArrayInt | None) -> bool:
    if xm is None:
        return False

    def unique(arr: ArrayInt) -> ArrayInt:
        return np.apply_along_axis(lambda x: np.unique(x).size, axis=1, arr=arr)

    rows = xm
    cols = xm.T
    sqrs = np.array([xm[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3].flatten() for i in range(3) for j in range(3)])
    catd = np.concatenate([rows, cols, sqrs], axis=0)
    return bool(np.all(unique(catd) == 9))


def parse_file_to_sudoku(filename: Path | str) -> list[ArrayInt]:
    """Reads Sudoku puzzles from a file.

    The file should contain one Sudoku puzzle per line, with the 9*9
    digits in a row. In this file, 0 means that the cell is not filled.
    Example:

    sudoku.txt:
    000075400000000008080190000300001060000000034000068170204000603900000020530200000
    300000000050703008000028070700000043000000000003904105400300800100040000968000200
    ...

    Args:
        filename (Path | str): The path to the file containing Sudoku puzzles.

    Returns:
        list[Grid]: A list of Sudoku grids.
    """
    sudokus: list[ArrayInt] = []
    with open(filename) as file:
        for line in file:
            line: str = line.strip()  # remove newline character
            nums: list[int] = list(map(int, line))
            if len(nums) == 81:  # 9*9 digits
                sudoku: ArrayInt = np.array(nums).reshape(9, 9).astype(Int)
                sudokus.append(sudoku)
    return sudokus
