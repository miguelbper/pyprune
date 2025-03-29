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
        # loop over grid
        for i in range(9):
            for j in range(9):
                # look at bitmask in the current cell
                c = bm[i, j]

                # if bitmask is empty,
                # meaning, no numbers are possible for that cell
                # reject by returning None
                if not c:
                    return None

                # if the bitmask is a singleton / power of two
                # meaning, that we know the number in that cell
                # then declare that cells in same square, row, col can't have the same num
                if c & (c - 1) == 0:
                    mask = ~c
                    u = (i // 3) * 3
                    v = (j // 3) * 3
                    bm[i, :] &= mask
                    bm[:, j] &= mask
                    bm[u : u + 3, v : v + 3] &= mask
                    bm[i, j] = c
        return bm


def sudoku_solver(sudoku: ArrayInt) -> ArrayInt | None:
    solver = Sudoku()

    # Convert the grid to a bitmask matrix
    # unknown cell -> bitmask of 1111111110, meaning all numbers are possible
    unknown = sum(1 << i for i in range(1, 10))
    bm = np.where(sudoku, 1 << sudoku, unknown)

    # Solve the grid (by providing the initial stack with just the bitmask matrix)
    sol = solver.solution([bm])
    return sol


def main() -> None:
    # Example sudoku
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
