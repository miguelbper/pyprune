"""Backtracking algorithm for solving constraint satisfaction problems.

Offers a Backtracking class that implements a backtracking algorithm.
The class is general purpose and can be used to solve any constraint
satisfaction puzzle. However, the class does not know the rules of any
specific problem. Users should inherit from this class and add the rules
of the problem.
"""

from collections.abc import Callable, Iterator

import numpy as np
from numba import njit
from numpy.typing import NDArray

Grid = NDArray[np.uint32]  # Each element is an int
Choices = NDArray[np.uint32]  # Each element is an int representing a set


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

    def solution(self) -> Grid | None:
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
    @njit
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
        return np.log2(cm).astype(np.uint32)

    @staticmethod
    @njit
    def accept(cm: Choices) -> np.bool:
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

        This function may optionally be overridden by the user to make
        more informed guesses for the list of possible choice matrices.
        To do this, implement expand in the child class. The input is
        the current choices matrix, and the output is a list of possible
        prunings of the choices matrix, such that for every solution of
        the problem, there exists a pruning in the list that contains
        the solution.

        Args:
            cm (Choices): The choices matrix to expand.

        Returns:
            list[Choices]: A list of new choice matrices, each
                representing a possible choice for the element with the
                fewest possible choices.
        """
        cardinality = np.sum((cm[..., None] & (1 << np.arange(32))) != 0, axis=-1)
        i, j = np.unravel_index(np.argmin(cardinality), cm.shape)
        powers_of_two = 1 << np.arange(32)
        powers_present = powers_of_two[cm[i, j] & powers_of_two > 0]
        cm_copies = np.repeat(cm[np.newaxis, ...], len(powers_present), axis=0)
        cm_copies[:, i, j] = powers_present
        return [cm_copies[k] for k in range(len(powers_present))]

    def prune(self, cm: Choices) -> Choices | None:
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
                cm_new = func(cm)
                if cm_new is None or not np.all(cm_new):
                    return None
                cm = cm_new
            prune_again = not np.array_equal(cm, cm_temp)
        return cm

    def rules(self) -> Iterator[Callable[[Choices], Choices | None]]:
        """Generator function that yields rule functions.

        Constraints are defined by the user as methods of a class that
        inherits from Backtracking. A rule is a method whose name starts
        with 'rule_', of type Callable[[Choices], Optional[Choices]].

        So, to define a rule, define a method in the child class as
        follows:

            def rule_my_rule(self, cm: Choices) -> Optional[Choices]:
                ...

        with the following properties (cm = input, om = output):
            - if the rule is violated, return None
            - For all i, j: om[i, j] ⊆ cm[i, j] (only remove elements)
            - If a filled grid which is a subset of cm satisfies the
              rule, then the filled grid is also a subset of om.

        Yields:
            Callable[[Choices], Optional[Choices]]: A function
                representing a rule of the problem.
        """
        for attr_name in dir(self):
            if attr_name.startswith("rule_"):
                attr_value = getattr(self, attr_name)
                if callable(attr_value):
                    yield attr_value
