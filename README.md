<div align="center">

# PyPrune
[![Python](https://img.shields.io/badge/Python-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-4dabcf?logo=numpy&logoColor=white)](https://numpy.org/)
[![Ruff](https://img.shields.io/badge/Ruff-261230?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/badge/uv-de5fe9?logo=uv&logoColor=white)](https://github.com/astral-sh/uv)
[![Code Quality](https://github.com/miguelbper/pyprune/actions/workflows/code-quality.yaml/badge.svg)](https://github.com/miguelbper/pyprune/actions/workflows/code-quality.yaml)
[![Unit Tests](https://github.com/miguelbper/pyprune/actions/workflows/tests.yaml/badge.svg)](https://github.com/miguelbper/pyprune/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/miguelbper/pyprune/graph/badge.svg)](https://codecov.io/gh/miguelbper/pyprune)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)

A simple interface to create fast solvers for constraint satisfaction problems

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

### Solver class

We will use Sudoku as an example of a problem that can be solved with PyPrune. We will walk through this [example](pyprune/sudoku/sudoku.py) file.

To create a Sudoku solver, we create a new class `Sudoku` which inherits from
`Backtracking`. By subclassing `Backtracking`, we get a generic backtracking
algorithm "for free". This means that we only need to implement the rules
specific to the puzzle, i.e. the logic of how to fill the puzzle "by hand".

The `Backtracking` class takes care of repeatedly using the rules to fill the
grid util it finds a mistake, solves the puzzle, or becomes blocked, and adapt
to each case:
- finds a mistake: rejects the current grid and backtracks to a previous grid
- solves the puzzle: returns the grid
- becomes blocked: make a guess on how to fill the grid

To implement the solver class, just add methods decorated with `rule` that
specify how you want to fill the grid.

```python
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
```

### Bit masks

In the example above, we use bit masks. PyPrune assumes that the puzzle we are trying to solve consists of filling numbers in a grid. This grid is represented as a numpy array. Each cell in this array consists of an integer bitmask with 1s on the bits of numbers that could be in that cell. Using as an example the Sudoku above, if a cell has the value $`(672)_{10} = (2^5 + 2^7 + 2^9)_{10} = (1010100000)_2`$, this means that in the Sudoku puzzle, the possible values for that cell are $`\{5,7,9\}`$.

### Solving the puzzle

Finally, to solve the puzzle, instantiate the solver with an initial condition and call `.solution()`.

```python
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

solver = Sudoku()

# Convert the grid to a bitmask matrix
# unknown cell -> bitmask of 1111111110, meaning all numbers are possible
unknown = sum(1 << i for i in range(1, 10))
bm = np.where(sudoku, 1 << sudoku, unknown)

# Solve the grid (by providing the initial stack with just the bitmask matrix)
sol = solver.solution([bm])
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
```

## Performance

The combination of **backtracking**, **bitmasks**, and representing the grids as **numpy arrays** (and using numpy array methods to do the computations on the grid) yields very fast algorithms for different puzzles. The speed [benchmark](pyprune/sudoku/benchmark.py) for sudokus shows that the algorithm above can solve 1000 sudokus in less than 2 seconds.

## Why not just use a SAT or CSP solver?

There exist excelent libraries like [Z3](https://github.com/Z3Prover/z3) and [OR-Tools](https://github.com/google/or-tools) that allow us to solve constraint satisfaction problems using a simple declarative language, while being exremely fast. I use PyPrune in scenarios where the problem/puzzle is not easy to express in the language offered by these libraries.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
