import numpy as np
from numpy.typing import NDArray
from typing import Optional, Iterator, Callable
from backtracking.subset import (
    elements,
    smallest,
    num_elements_numba,
)
from numba import njit

Grid = NDArray[np.uint32]     # Each element is an int
Choices = NDArray[np.uint32]  # Each element is an int representing a set


def grid(cm: Choices) -> Grid:
    """Convert from a choices matrix to a grid.

    Assumes that no element of cm is zero. When used in
    'solution_generator', this is true because of the 'accept' function.

    Parameters:
        cm (Choices): The input choices matrix.

    Returns:
        Grid: The resulting grid.
    """
    return np.vectorize(smallest)(cm)


@njit
def accept(cm: Choices) -> bool:
    """Checks if all elements of the choice matrix are singletons.

    Parameters:
        cm (Choices): The choice matrix to be checked.

    Returns:
        bool: True if all elements of the choice matrix are singletons.
    """
    return np.all(np.logical_and(cm, cm & (cm - 1) == 0))


@njit
def argmin_num_elements(cm: Choices) -> tuple[int, int]:
    """Finds i, j that minimizes the number of elements in cm[i, j],
    subject to the condition that cm[i, j] has at least two elements.

    This function is called inside expand.

    Args:
        cm (Choices): The Choices matrix.

    Returns:
        tuple[int, int]: The indices (i, j).
    """
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


class Backtracking:
    """
    Represents a backtracking problem.

    Usage:
        - Define new class that inherits from this class.
        - Override __init__ to specify problem constants.
        - Define the constraints as methods of the new class.
        - Instantiate by providing initial choices matrix.
        - Call 'solution' or 'solutions' to find the solution(s).

    Attributes:
        cm0 (Choices): The initial matrix of choices.

    Public methods (meant to be called by the user):
        __init__(self, cm0: Choices) -> None:
            Initializes a Backtracking object.

        solution(self) -> Optional[Grid]:
            Finds a solution using the backtracking algorithm.

        solutions(self) -> list[Grid]:
            Returns a list of all possible solutions for the problem.

        constraint(self) -> Callable[[Choices], Optional[Choices]]:
            Generator function that yields constraint functions.

    Private methods (only meant to be called by the class):
        solution_generator(self) -> Iterator[Grid]:
            Generates solutions using the backtracking algorithm.

        prune(self, cm: Choices) -> Optional[Choices]:
            Prunes the choices based on the constraints.
    """

    def __init__(self, cm0: Choices) -> None:
        """Initializes a Backtracking object.

        Args:
            cm0 (Choices): The initial matrix of choices.

        Returns:
            None
        """
        self.cm0 = cm0.astype(np.uint32)

    # TODO: integration test
    def solution_generator(self) -> Iterator[Grid]:
        """Generates solutions using backtracking algorithm.

        Generator that is called from the 'solution' and 'solutions'
        methods. A generator is used to prevent code duplication.

        Yields:
            Grid: A valid solution grid.
        """
        stack = [self.cm0]
        while stack:
            cm = self.prune(stack.pop())
            if cm is None:
                continue
            if accept(cm):
                yield grid(cm)
            stack += self.expand(cm)

    def solution(self) -> Optional[Grid]:
        """Finds a solution using a backtracking algorithm.

        Returns:
            Optional[Grid]: The solution grid if found, None otherwise.
        """
        return next(self.solution_generator(), None)

    def solutions(self) -> list[Grid]:
        """Returns a list of all possible solutions for the problem.

        Returns:
            A list of Grid objects representing the possible solutions.
        """
        return list(self.solution_generator())

    # TODO: test overriding expand
    @staticmethod
    @njit
    def expand(cm: Choices) -> list[Choices]:
        """Expands the given choices matrix by selecting the element with
        the fewest possible choices, and creating new choice matrices for
        each possible choice of that element.

        Args:
            cm (Choices): The choices matrix to expand.

        Returns:
            list[Choices]: A list of new choice matrices, each representing
                a possible choice for the element with the fewest possible
                choices.
        """
        i, j = argmin_num_elements(cm)
        if i == -1:
            return [cm for _ in range(0)]  # empty list that numba can infer
        ans = []
        for x in elements(cm[i, j]):
            cmx = np.copy(cm)
            cmx[i, j] &= 1 << x  # cmx[i, j] = remove_except(cmx[i, j], x)
            ans.append(cmx)
        return ans

    def prune(self, cm: Choices) -> Optional[Choices]:
        """Prunes the choices based on the constraints.

        Loops through the constraints defined by the users to either
        reject a choices matrix or prune it.

        Args:
            cm (Choices): The choices to be pruned.

        Returns:
            Optional[Choices]: The pruned choices, or None if the
                constraints are violated.
        """
        prune_again = True
        while prune_again:
            cm_temp = np.copy(cm)
            for func in self.constraints():
                cm = func(cm)
                if cm is None or not np.all(cm):
                    return None
            prune_again = not np.array_equal(cm, cm_temp)
        return cm

    def constraints(self) -> Iterator[Callable[[Choices], Optional[Choices]]]:
        """Generator function that yields constraint functions.

        Constraints are defined by the user as methods of a class
        that inherits from Backtracking. A constraint is a method
        whose name starts with 'constraint_', of type
        Callable[[Choices], Optional[Choices]].

        Yields:
            Callable[[Choices], Optional[Choices]]: A function
                representing a constraint of the problem.
        """
        for attr_name in dir(self):
            if attr_name.startswith('constraint_'):
                attr_value = getattr(self, attr_name)
                if callable(attr_value):
                    yield attr_value
