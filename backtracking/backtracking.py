import numpy as np
from typing import Optional, Iterator
from copy import deepcopy
from subset import remove_except, elements, smallest
from numba import njit

Grid = np.ndarray
Choices = np.ndarray


class Problem:
    def __init__(self, cm0: Choices) -> None:
        self.constraints = []
        self.cm0 = cm0

    def constraint(self, func):
        self.constraints.append(func)
        return func

    # TODO: unit test
    def grid(self, cm: Choices) -> Grid:
        return np.vectorize(smallest)(cm)

    @njit
    def solution_generator(self) -> Iterator[Grid]:
        stack = [self.cm0]
        while stack:
            cm = self.prune(stack.pop())
            if not cm:
                continue
            if self.accept(cm):
                yield self.grid(cm)
            stack += self.expand(cm)

    def solution(self) -> Optional[Grid]:
        return next(self.solution_generator(), None)

    def solutions(self) -> list[Grid]:
        return list(self.solution_generator())

    # TODO: unit test
    @njit
    def accept(self, cm: Choices) -> bool:
        return np.all(np.logical_and(cm, cm & (cm - 1) == 0))

    # TODO: unit test
    @njit
    def expand(self, cm: Choices) -> list[Choices]:
        # find the cell with the fewest choices (assumes cm[i][j] < 128)
        m, n = cm.shape
        bits = np.unpackbits(cm.astype(np.uint8)).reshape(m, n, 8)
        num_elements = np.sum(bits, axis=2)
        indx = np.argmin(num_elements)
        i, j = np.unravel_index(indx, cm.shape)

        # create a new grid for each possible choice
        ans = []
        for x in elements(cm[i, j]):
            cmx = deepcopy(cm)
            cmx[i, j] = remove_except(cmx[i, j], x)
            ans.append(cmx)
        return ans

    @njit
    def prune(self, cm: Choices) -> Optional[Choices]:
        prune_again = True
        while prune_again:
            cm_temp = np.copy(cm)
            for func in self.constraints:
                cm = func(cm)
                if not (cm and np.all(cm)):
                    return None
            prune_again = not np.array_equal(cm, cm_temp)
        return cm
