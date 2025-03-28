<div align="center">

# PyPrune
[![Python](https://img.shields.io/badge/Python-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-4dabcf?logo=numpy&logoColor=white)](https://numpy.org/)
[![Ruff](https://img.shields.io/badge/Ruff-261230?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/badge/uv-de5fe9?logo=uv&logoColor=white)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)
[![Code Quality](https://github.com/miguelbper/pyprune/actions/workflows/code-quality.yaml/badge.svg)](https://github.com/miguelbper/pyprune/actions/workflows/code-quality.yaml)
[![Unit Tests](https://github.com/miguelbper/pyprune/actions/workflows/tests.yaml/badge.svg)](https://github.com/miguelbper/pyprune/actions/workflows/tests.yaml)

A simple interface to create solvers for constraint satisfaction problems

</div>

---

## Description

Offers a `Backtracking` class that implements a backtracking algorithm. The class is general purpose and can be used to solve a general constraint satisfaction puzzle in a grid. Users should inherit from this class and add the rules of the problem.

## Installation

```bash
# Install from PyPi
pip install pyprune
```

## Usage

We will use Sudoku as an example of a problem that can be solved with PyPrune.

To create a Sudoku solver, we create a new class `Sudoku` which inherits from `Backtracking`. By inheritance, we get a generic backtracking algorithm. We only need to implement the rules specific to the puzzle. In PyPrune, this is done by implementing functions decorated with `rule`. A `rule` is the sequence of operations that we would do on the grid when filling the puzzle by hand.

```python
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

        # loop over grid
        for i in range(n):
            for j in range(n):

                # look at bitmask in the current cell
                c = bm[i, j]

                # if bitmask is empty,
                # meaning no numbers are possible for that cell
                # reject by returning None
                if not c:
                    return None

                # if the bitmask is a singleton / power of two
                # meaning that we know the number in that cell
                # then declare that cells in same square, row, col can't have the same num
                if c & (c - 1) == 0:
                    mask = ~c
                    u = (i // k) * k
                    v = (j // k) * k
                    bm[i, :] &= mask
                    bm[:, j] &= mask
                    bm[u : u + k, v : v + k] &= mask
                    bm[i, j] = c
        return bm
```

In the example above, we use bit masks. PyPrune assumes that the puzzle we are trying to solve consists of filling numbers in a grid. This grid is represented as a numpy array. Each cell in this array consists of an integer bitmask with 1s on the bits of numbers that could be in that cell. Using as an example the Sudoku above, if a cell has the value $`(672)_{10} = (2^5 + 2^7 + 2^9)_{10} = (1010100000)_2`$, this means that in the Sudoku puzzle, the possible values for that cell are $`\{5,7,9\}`$.

Finally, to solve the puzzle, instantiate the solver with an initial condition and call `.solution()`.

```python
solver = Sudoku(3)

xm = sudoku                          # xm = sudoku is a given array with some cells already filled
bm = np.where(xm, 1 << xm, unknown)  # bm is the corresponding bitmask matrix
sol = solver.solution([bm])          # call the solver and get the solution
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
