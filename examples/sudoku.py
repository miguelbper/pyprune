import os

import numpy as np
from numba import njit

from pyprune.backtracking import Backtracking, Choices, Grid


# The class that should be implemented to solve a Sudoku puzzle
class Sudoku(Backtracking):
    """A class representing the Sudoku puzzle solver.

    Inherits from the `Backtracking` class.

    Attributes:
        cm (Choices): The initial choices matrix representing the grid.

    Methods:
        rule_sudoku(cm: Choices) -> Optional[Choices]: Applies the rules
            of Sudoku to the choices matrix.
    """

    @staticmethod
    @njit
    def rule_sudoku(cm: Choices) -> Choices | None:
        """Applies the rules of Sudoku.

        If a cell (i, j) has value x, then
        - remove x from the other cells in the same row
        - remove x from the other cells in the same column
        - remove x from the other cells in the same box

        Args:
            cm (Choices): The choices matrix representing the grid.

        Returns:
            Optional[Choices]: The updated choices matrix after applying
                the Sudoku rules.
        """
        cm = np.copy(cm)
        for i in range(9):
            for j in range(9):
                c = cm[i, j]
                if c and c & (c - 1) == 0:
                    mask = ~c
                    u = (i // 3) * 3
                    v = (j // 3) * 3
                    cm[i, :] &= mask
                    cm[:, j] &= mask
                    cm[u : u + 3, v : v + 3] &= mask
                    cm[i, j] = c
        return cm


# This is not really necessary, but we implement it just to sanity check
# that a solution obtained by Sudoku class is correct
def is_sudoku_solution(cm: Grid) -> bool:
    """Check if the given Sudoku grid is a valid solution.

    Args:
        cm (Grid): The Sudoku grid to check.

    Returns:
        bool: True if the grid is a valid solution, False otherwise.
    """
    # Check rows
    for i in range(9):
        if len(np.unique(cm[i, :])) != 9:
            return False

    # Check columns
    for j in range(9):
        if len(np.unique(cm[:, j])) != 9:
            return False

    # Check boxes
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            box = cm[i : i + 3, j : j + 3]
            if len(np.unique(box)) != 9:
                return False

    return True


# Implemented so that we can read the Sudoku puzzles from the file
def parse_file_to_sudoku(filename: str) -> list[Grid]:
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
                sudokus.append(sudoku.astype(np.uint32))
    return sudokus


# Example usage
# ----------------------------------------------------------------------

if __name__ == "main":
    # get a sudoku puzzle from the file
    sudoku = parse_file_to_sudoku(os.path.join("examples", "sudoku.txt"))[0]
    print("\nsudoku = \n", sudoku)
    # sudoku =
    # [[0 0 0 0 7 5 4 0 0]
    #  [0 0 0 0 0 0 0 0 8]
    #  [0 8 0 1 9 0 0 0 0]
    #  [3 0 0 0 0 1 0 6 0]
    #  [0 0 0 0 0 0 0 3 4]
    #  [0 0 0 0 6 8 1 7 0]
    #  [2 0 4 0 0 0 6 0 3]
    #  [9 0 0 0 0 0 0 2 0]
    #  [5 3 0 2 0 0 0 0 0]]

    # define the choices matrix
    # if cell (i, j) is filled with value x, then cm[i, j] = 1 << x
    # if cell (i, j) is empty, then cm[i, j] = 0b1111111110 = 2**10 - 2 = 1022
    # in more complicated cases, can use subset.subset([x1, x2, ..., xk])
    cm = np.where(sudoku, 2**sudoku, (2**10 - 2) * np.ones((9, 9))).astype(np.uint32)
    print("\ncm = \n", cm)
    # Cm = [[1022 1022 1022 1022  128   32   16 1022 1022] [1022 1022 1022 1022
    # 1022 1022 1022 1022  256] [1022  256 1022    2  512 1022 1022 1022 1022] [   8
    # 1022 1022 1022 1022    2 1022   64 1022] [1022 1022 1022 1022 1022 1022 1022 8
    # 16] [1022 1022 1022 1022   64  256    2  128 1022] [   4 1022   16 1022 1022
    # 1022   64 1022    8] [ 512 1022 1022 1022 1022 1022 1022    4 1022] [  32 8
    # 1022    4 1022 1022 1022 1022 1022]]

    # create a Sudoku object / problem
    problem = Sudoku(cm)

    # find solutions to the problem
    solutions = problem.solutions()
    n_solutions = len(solutions)
    solution = solutions[0]
    correct = is_sudoku_solution(solution)
    print(f"\n{n_solutions = }")
    print(f"{correct = }")
    print("solution = \n", solution)
    n_solutions = 1
    correct = True
    # solution =
    # [[6 9 3 8 7 5 4 1 2]
    #  [1 4 5 6 3 2 7 9 8]
    #  [7 8 2 1 9 4 3 5 6]
    #  [3 5 7 4 2 1 8 6 9]
    #  [8 1 6 9 5 7 2 3 4]
    #  [4 2 9 3 6 8 1 7 5]
    #  [2 7 4 5 1 9 6 8 3]
    #  [9 6 8 7 4 3 5 2 1]
    #  [5 3 1 2 8 6 9 4 7]]
