# PyPrune
Backtracking algorithm for solving constraint satisfaction problems.

Offers a `Backtracking` class that implements a backtracking algorithm. The class is general purpose and can be used to solve a general constraint satisfaction puzzle in a grid. Users should inherit from this class and add the rules of the problem.

## Explanation
We will use Sudoku as an example to illustrate the usage of this package. If we are given a Sudoku puzzle
```python
sudoku = [
    [0, 0, 0, 0, 7, 5, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 8],
    [0, 8, 0, 1, 9, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 1, 0, 6, 0],
    [0, 0, 0, 0, 0, 0, 0, 3, 4],
    [0, 0, 0, 0, 6, 8, 1, 7, 0],
    [2, 0, 4, 0, 0, 0, 6, 0, 3],
    [9, 0, 0, 0, 0, 0, 0, 2, 0],
    [5, 3, 0, 2, 0, 0, 0, 0, 0],
]
```
where `0` means that the cell is unknown, we represent the puzzle by a **choices matrix** `cm`. This is a matrix where each cell contains the set of elements that could possibly be in that cell. For example, `cm[0, 0] = {1, 2, 3, 4, 5, 6, 7, 8, 9}` and `cm[0, 4] = {7}` (actually, `cm[i, j]` is an `int` representing the set, see an explanation of this below). Throughout the backtracking algorithm (offered by the `Backtracking` class), each cell in this choices matrix gets pruned, until each cell contains only one value.

The user has to implement a class representing the problem that inherits from `Backtracking`. In this class, the user should add the rules of how should cells be pruned, i.e. the rules of the specific puzzle being solved.

Strictly speaking, `cm[i, j]` is not a set but an integer representing that set. The binary representation of the integer has 1s at the indices of the elements in the set. For example, the integer `5` has binary representation `101`, which means that the set `{0, 2}` is represented by `5`. The advantage of this is that we can perform the required set operations with bit tricks, which is much faster than using Python sets.

## Example usage

See `pyprune/examples/sudoku.py`.

**Step 1.** Implement a class with the rules of the puzzle. To do so, implement a class inheriting from `Backtracking`. Override `prune` and optionally `expand`.
```python
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
        cm = np.where(sudoku, 1 << sudoku, (2**10 - 2) * np.ones((9, 9), dtype=np.int32))
        self.stack = [cm]

    @rule
    def sudoku(cm: Choices) -> Choices | None:
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
```

**Step 2.** Instantiate problem class.
```python
problem = Sudoku(sudoku)
```

**Step 3.** Find solutions of the problem. Here is when the backtracking algorithm is called.
```python
solution = problem.solution()
print(solution)
'''
solution =
[[6 9 3 8 7 5 4 1 2]
 [1 4 5 6 3 2 7 9 8]
 [7 8 2 1 9 4 3 5 6]
 [3 5 7 4 2 1 8 6 9]
 [8 1 6 9 5 7 2 3 4]
 [4 2 9 3 6 8 1 7 5]
 [2 7 4 5 1 9 6 8 3]
 [9 6 8 7 4 3 5 2 1]
 [5 3 1 2 8 6 9 4 7]]
'''
```
