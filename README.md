<div align="center">

# pyprune
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

This package offers a `Backtracking` class that implements a backtracking algorithm. The class is general purpose and can be used to solve a general constraint satisfaction puzzle in a grid. Users should inherit from this class and add the rules of the problem. The package is tiny, has NumPy as its only dependency, and can be used as a learning resource if you want to learn how to implement an algorithm like this yourself.

## Installation

```bash
# Install from PyPi
pip install pyprune
```

## Usage

### Solver class

We will use Sudoku as an example of a problem that can be solved with pyprune. We will walk through this [example](pyprune/sudoku/sudoku.py) file.

To create a Sudoku solver, we create a new class `Sudoku` which inherits from
`Backtracking`. By subclassing `Backtracking`, we get a generic backtracking
algorithm for free. This means that we only need to implement the rules
specific to the puzzle, i.e. the logic of how to fill the puzzle as if "by hand".

The `Backtracking` class takes care of repeatedly using the rules to fill the
grid until it finds a mistake, solves the puzzle, or becomes blocked, and adapts
to each case:
- If it solves the puzzle, it returns the grid
- If it finds a mistake, it rejects the current grid and backtracks to a previous grid
- If it becomes blocked, it chooses a cell and guesses how to fill that cell

To implement the solver class, just add methods decorated with `rule` that
specify how you want to fill the grid.

```python
import numpy as np

from pyprune.backtracking import ArrayBitMask, ArrayInt, Backtracking, rule


class Sudoku(Backtracking):
    @rule
    def sudoku(self, bm: ArrayBitMask) -> ArrayBitMask | None:
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
```

That's it! The class above is a fully-fledged sudoku solver. We will just need to instantiate it and then call the solver for a specific puzzle we want to solve.

### Bit masks

In the example above, we use bit masks. pyprune assumes that the puzzle we are trying to solve consists of filling numbers in a grid. This grid is represented as a NumPy array. Each cell in this array consists of an integer bitmask with 1s on the bits of numbers that could be in that cell. Using as an example the Sudoku above, if a cell has the value $`(672)_{10} = (2^5 + 2^7 + 2^9)_{10} = (1010100000)_2`$, this means that in the Sudoku puzzle, the possible values for that cell are $`\{5,7,9\}`$.

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
unknown = sum(1 << i for i in range(1, 10))  # (1111111110)_2 - all numbers are possible
bm = np.where(sudoku, 1 << sudoku, unknown)  # array of bitmasks for initial grid
sol = solver.solution([bm])  # input to solver is the initial stack of ArrayBitMask
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

The combination of backtracking, bitmasks, and representing the grids as NumPy arrays yields very fast algorithms for different puzzles. The speed [benchmark](pyprune/sudoku/benchmark.py) for sudokus shows that the algorithm above can solve **1000 sudokus in less than 2 seconds**.

## Why not just use a SAT or CSP solver?

There exist excellent libraries like [Z3](https://github.com/Z3Prover/z3) and [OR-Tools](https://github.com/google/or-tools) that allow us to solve constraint satisfaction problems using a simple declarative language. Moreover, these solvers are extremelly fast. I use pyprune in scenarios where the problem/puzzle is not easy to express in the language offered by these libraries.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
