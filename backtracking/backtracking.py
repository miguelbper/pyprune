import numpy as np
from numpy.typing import NDArray
from typing import Optional, Iterator
from backtracking.subset import (
    elements,
    smallest_numba,
    num_elements_numba,
)
from numba import njit

Grid = NDArray[np.uint32]
Choices = NDArray[np.uint32]


def grid(cm: Choices) -> Grid:
    return np.vectorize(smallest_numba)(cm)


@njit
def accept(cm: Choices) -> bool:
    return np.all(np.logical_and(cm, cm & (cm - 1) == 0))


@njit
def argmin_num_elements(cm: Choices) -> tuple[int, int]:
    m, n = cm.shape
    min_i, min_j = -1, -1
    min_num_elements = np.inf
    for i in range(m):
        for j in range(n):
            n_elements = num_elements_numba(cm[i, j])
            if n_elements == 2:
                return i, j
            if 1 < n_elements < min_num_elements:
                min_i, min_j = i, j
                min_num_elements = n_elements
    return min_i, min_j


@njit
def expand(cm: Choices) -> list[Choices]:
    i, j = argmin_num_elements(cm)
    ans = []
    for x in elements(cm[i, j]):
        cmx = np.copy(cm)
        cmx[i, j] &= 1 << x  # cmx[i, j] = remove_except(cmx[i, j], x)
        ans.append(cmx)
    return ans


class Problem:
    def __init__(self, cm0: Choices) -> None:
        self.constraints = []
        self.cm0 = cm0.astype(np.uint32)

    # TODO: integration test
    def solution_generator(self) -> Iterator[Grid]:
        stack = [self.cm0]
        while stack:
            cm = self.prune(stack.pop())
            if not cm:
                continue
            if accept(cm):
                yield grid(cm)
            stack += expand(cm)

    def solution(self) -> Optional[Grid]:
        return next(self.solution_generator(), None)

    def solutions(self) -> list[Grid]:
        return list(self.solution_generator())

    def constraint(self, func):
        self.constraints.append(func)
        return func

    # TODO: integration test
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
