from typing import Optional, Iterator, Callable
import numpy as np
from numpy.typing import NDArray
from numba import njit
from backtracking.subset import elements, smallest, num_elements_numba

Grid = NDArray[np.uint32]     # Each element is an int
Choices = NDArray[np.uint32]  # Each element is an int representing a set


@njit
def argmin_num_elements(cm: Choices) -> tuple[int, int]:
    """Finds i, j that minimizes the number of elements in cm[i, j],
    subject to the condition that cm[i, j] has at least two elements.

    If no cell has at least two elements, then (-1, -1) is returned.
    This function is called inside expand. In that case, it's guaranteed
    that there is at least one cell with at least two elements.

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
    """Represents a backtracking problem.

    Usage:
        - Define new class that inherits from this class.
        - Override __init__ to specify problem constants.
        - Define the rules as methods of the new class.
        - Instantiate by providing initial choices matrix.
        - Call 'solution' or 'solutions' to find the solution(s).

    Attributes:
        cm (Choices): The initial matrix of choices.

    Public methods (meant to be called by the user):
        __init__(self, cm: Choices) -> None:
            Initializes a Backtracking object.

        solution(self) -> Optional[Grid]:
            Finds a solution using the backtracking algorithm.

        solutions(self) -> list[Grid]:
            Returns a list of all possible solutions for the problem.

    Private methods (only meant to be called by the class):
        solution_generator(self) -> Iterator[Grid]:
            Generates solutions using the backtracking algorithm.

        grid(cm: Choices) -> Grid:
            Converts from a choices matrix to a grid.

        accept(self, cm: Choices) -> bool:
            Checks if all elements of the choice matrix are singletons.

        expand(self, cm: Choices) -> list[Choices]:
            Chooses a cell and lists the possible values for that cell.

        prune(self, cm: Choices) -> Optional[Choices]:
            Prunes the choices based on the rules.

        rules(self) -> Callable[[Choices], Optional[Choices]]:
            Generator function that yields rule functions.
    """

    def __init__(self, cm: Choices) -> None:
        """Initializes a Backtracking object.

        Args:
            cm (Choices): The initial matrix of choices.

        Returns:
            None
        """
        self.cm = cm.astype(np.uint32)

    def solution_generator(self) -> Iterator[Grid]:
        """Generates solutions using backtracking algorithm.

        Generator that is called from the 'solution' and 'solutions'
        methods. A generator is used to prevent code duplication.

        Yields:
            Grid: A valid solution grid.
        """
        stack = [self.cm]
        while stack:
            cm = self.prune(stack.pop())
            if cm is None:
                continue
            if self.accept(cm):
                yield self.grid(cm)
            else:
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

    @staticmethod
    def grid(cm: Choices) -> Grid:
        """Convert from a choices matrix to a grid.

        Assumes that no element of cm is zero. When used in
        'solution_generator', this is true because of the 'accept'
        function.

        Parameters:
            cm (Choices): The input choices matrix.

        Returns:
            Grid: The resulting grid.
        """
        return np.vectorize(smallest)(cm).astype(np.uint32)

    @staticmethod
    @njit
    def accept(cm: Choices) -> bool:
        """Checks if all elements of the choice matrix are singletons.

        Parameters:
            cm (Choices): The choice matrix to be checked.

        Returns:
            bool: True if all elements of cm are singletons.
        """
        return np.all(np.logical_and(cm, cm & (cm - 1) == 0))

    @staticmethod
    @njit
    def expand(cm: Choices) -> list[Choices]:
        """Chooses a cell and lists the possible values for that cell.

        Expands the given choices matrix by selecting the element with
        the fewest possible choices, and creating new choice matrices
        for each possible choice of that element.

        Args:
            cm (Choices): The choices matrix to expand.

        Returns:
            list[Choices]: A list of new choice matrices, each
                representing a possible choice for the element with the
                fewest possible choices.
        """
        i, j = argmin_num_elements(cm)
        ans = []
        for x in elements(cm[i, j]):
            cmx = np.copy(cm)
            cmx[i, j] &= 1 << x  # cmx[i, j] = remove_except(cmx[i, j], x)
            ans.append(cmx)
        return ans

    def prune(self, cm: Choices) -> Optional[Choices]:
        """Prunes the choices based on the rules.

        Loops through the rules defined by the users to either reject a
        choices matrix or prune it.

        Args:
            cm (Choices): The choices to be pruned.

        Returns:
            Optional[Choices]: The pruned choices, or None if the rules
                are violated.
        """
        prune_again = True
        while prune_again:
            cm_temp = np.copy(cm)
            for func in self.rules():
                cm = func(cm)
                if cm is None or not np.all(cm):
                    return None
            prune_again = not np.array_equal(cm, cm_temp)
        return cm

    def rules(self) -> Iterator[Callable[[Choices], Optional[Choices]]]:
        """Generator function that yields rule functions.

        Constraints are defined by the user as methods of a class that
        inherits from Backtracking. A rule is a method whose name starts
        with 'rule_', of type Callable[[Choices], Optional[Choices]].

        Yields:
            Callable[[Choices], Optional[Choices]]: A function
                representing a rule of the problem.
        """
        for attr_name in dir(self):
            if attr_name.startswith('rule_'):
                attr_value = getattr(self, attr_name)
                if callable(attr_value):
                    yield attr_value
