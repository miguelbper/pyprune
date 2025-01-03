import os

import numpy as np

from pyprune.backtracking import Backtracking, Choices, Grid, rule


# The class that should be implemented to solve a Sudoku puzzle
class Sudoku(Backtracking):
    """A class representing the Sudoku puzzle solver.

    Inherits from the `Backtracking` class.

    Attributes:
        cm (Choices): The initial choices matrix representing the grid.

    Methods:
        prune(cm: Choices) -> Choices | None: Applies the rules
            of Sudoku to the choices matrix.
    """

    def __init__(self, sudoku: Grid) -> None:
        super().__init__()
        cm = np.where(sudoku, 1 << sudoku, (2**10 - 2) * np.ones((9, 9), dtype=np.int32))
        self.stack = [cm]

        I, J, K, L = np.indices((9, 9, 9, 9))  # noqa: E741
        same_row = I == K
        same_col = J == L
        same_reg = (I // 3 == K // 3) & (J // 3 == L // 3)
        different = ~(same_row & same_col)
        self.neigh = different & (same_reg | same_row | same_col)

    @rule
    def sudoku(self, cm: Choices) -> Choices | None:
        """Applies the rules of Sudoku.

        If a cell (i, j) has value x, then
        - remove x from the other cells in the same row
        - remove x from the other cells in the same column
        - remove x from the other cells in the same box

        Args:
            cm (Choices): The choices matrix representing the grid.

        Returns:
            Choices | None: The updated choices matrix after applying
                the Sudoku rules.
        """
        nums = (1 << np.arange(1, 10)).reshape(-1, 1, 1)  # (9, 1, 1) [d, ...]
        cm_eq_nums = cm == nums  # (9, 9, 9) [d, i, j]
        masks = nums * np.any(cm_eq_nums.reshape(-1, 1, 1, 9, 9) & self.neigh, axis=(3, 4))  # noqa: E501 (9, 9, 9) [d, i, j]
        mask = np.sum(masks, axis=0)  # (9, 9) [i, j]
        return cm & ~mask


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
                sudokus.append(sudoku.astype(np.int32))
    return sudokus


# Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # get a sudoku puzzle from the file
    sudoku = parse_file_to_sudoku(os.path.join("pyprune", "examples", "sudoku.txt"))[0]
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

    # create a Sudoku object / problem
    problem = Sudoku(sudoku)

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
