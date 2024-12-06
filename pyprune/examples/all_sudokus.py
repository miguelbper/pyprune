import os

import numpy as np
from tqdm import tqdm

from pyprune.examples.sudoku import Sudoku, parse_file_to_sudoku

sudokus = parse_file_to_sudoku(os.path.join("pyprune", "examples", "sudoku.txt"))

for sudoku in tqdm(sudokus):
    cm = np.where(sudoku, 2**sudoku, (2**10 - 2) * np.ones((9, 9))).astype(np.uint32)
    problem = Sudoku(cm)
    solutions = problem.solutions()
