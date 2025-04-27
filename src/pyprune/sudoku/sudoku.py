import numpy as np

from pyprune.backtracking import ArrayBitMask, ArrayInt, Backtracking, rule


class Sudoku(Backtracking):
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
        for (i, j), b in np.ndenumerate(bm):
            if not b:  # if the cell is empty, reject grid
                return None
            if b & (b - 1) == 0:  # if only one number is possible in this cell...
                mask = ~b
                box_i = (i // 3) * 3
                box_j = (j // 3) * 3
                bm[i, :] &= mask  # update the row
                bm[:, j] &= mask  # update the column
                bm[box_i : box_i + 3, box_j : box_j + 3] &= mask  # update the box
                bm[i, j] = b  # reset the cell to the initial bitmask
        return bm


def sudoku_solver(sudoku: ArrayInt) -> ArrayInt | None:
    """Solves a Sudoku puzzle.

    Args:
        sudoku (ArrayInt): The initial Sudoku grid.

    Returns:
        ArrayInt | None: The solution grid if found, None otherwise.
    """
    solver = Sudoku()
    unknown = sum(1 << i for i in range(1, 10))  # (1111111110)_2 - all numbers are possible
    bm = np.where(sudoku, 1 << sudoku, unknown)  # array of bitmasks for initial grid
    return solver.solution([bm])  # input to solver is the initial stack of ArrayBitMask


def main() -> None:
    sudoku = np.array(
        [
            [0, 0, 0, 0, 7, 5, 4, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 8],
            [0, 8, 0, 1, 9, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 1, 0, 6, 0],
            [0, 0, 0, 0, 0, 0, 0, 3, 4],
            [0, 0, 0, 0, 6, 8, 1, 7, 0],
            [2, 0, 4, 0, 0, 0, 6, 0, 3],
            [9, 0, 0, 0, 0, 0, 0, 2, 0],
            [5, 3, 0, 2, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    solution = sudoku_solver(sudoku)
    print(solution)
    """
    [[6 9 3 8 7 5 4 1 2]
     [1 4 5 6 3 2 7 9 8]
     [7 8 2 1 9 4 3 5 6]
     [3 5 7 4 2 1 8 6 9]
     [8 1 6 9 5 7 2 3 4]
     [4 2 9 3 6 8 1 7 5]
     [2 7 4 5 1 9 6 8 3]
     [9 6 8 7 4 3 5 2 1]
     [5 3 1 2 8 6 9 4 7]]
    """


if __name__ == "__main__":
    main()
