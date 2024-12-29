"""Backtracking algorithm for solving constraint satisfaction problems.

Offers a Backtracking class that implements a backtracking algorithm.
The class is general purpose and can be used to solve any constraint
satisfaction puzzle. However, the class does not know the rules of any
specific problem. Users should inherit from this class and add the rules
of the problem.
"""

import inspect
from collections.abc import Callable, Iterator
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

Grid = NDArray[np.int32]  # Each element is an int
Choices = NDArray[np.int32]  # Each element is an int representing a set


def rule(func: Callable) -> Callable:
    func.rule = True
    return func


class Backtracking:
    """Represents a backtracking problem.

    Usage:
        1. Define a new class that inherits from this class.

        2. __init__:
            - Override
            - Do super().__init__()
            - Define the initial stack

        3. expand -> expand_cell
            Options (from less to more "manual")
            - Leave the methods as is / do nothing
            - Override expand_cell to specify what cell should be chosen
            - Override expand to specify different logic

        4. prune_repeatedly -> prune -> @rule's
            Options (from less to more "manual")
            - Define methods decorated with @rule, they will be called by prune
            - Override prune
            - Override prune_repeatedly

        5. Instantiate and call 'solution' or 'solutions' to find the solution(s).

    Attributes:
        cm (Choices): The initial matrix of choices.

    Public methods (meant to be called by the user):
        __init__(self, cm: Choices) -> None:
            Initializes a Backtracking object.

        solution(self) -> Grid | None:
            Finds a solution using the backtracking algorithm.

        solutions(self) -> list[Grid]:
            Returns a list of all possible solutions for the problem.

    Private methods (only meant to be called by the class):
        solution_generator(self) -> Iterator[Grid]:
            Generates solutions using the backtracking algorithm.

        grid(cm: Choices) -> Grid:
            Converts from a choices matrix to a grid.

        reject(cm: Choices | None) -> bool:
            True if cm is None or if cm has an empty cell.

        accept(cm: Choices) -> bool:
            Checks if all elements of the choice matrix are singletons.

        expand(self, cm: Choices) -> list[Choices]:
            Chooses a cell and lists the possible values for that cell.
            Can optionally be overriden by the user.

        prune_repeatedly(self, cm: Choices) -> Choices | None:
            Repeatedly calls prune, until cm is no longer changed.

        prune(self, cm: Choices) -> Choices | None:
            Defines the rules of the problem. Should be implemented by
            the user.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initializes a Backtracking object. Should create self.stack
        attribute.

        Args:
            Whatever you want to pass as argument.

        Returns:
            None
        """
        self.rules = self.get_rules()

    def solution_generator(self) -> Iterator[Grid]:
        """Generates solutions using backtracking algorithm.

        Generator that is called from the 'solution' and 'solutions'
        methods. A generator is used to prevent code duplication.

        Yields:
            Grid: A valid solution grid.
        """
        stack = deepcopy(self.stack)
        while stack:
            cm = self.prune_repeatedly(stack.pop())
            if cm is None:
                continue
            if self.accept(cm):
                yield self.grid(cm)
            else:
                stack += self.expand(cm)

    def solution(self) -> Grid | None:
        """Finds a solution using a backtracking algorithm.

        Returns:
            Grid | None: The solution grid if found, None otherwise.
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

        Assumes that all elements of cm are singletons. When used in
        'solution_generator', this is true because of the 'accept'
        function.

        Args:
            cm (Choices): The input choices matrix.

        Returns:
            Grid: The resulting grid.
        """
        return np.log2(cm).astype(np.int32)

    @staticmethod
    def reject(cm: Choices | None) -> bool:
        """Checks if the choice matrix is invalid.

        Args:
            cm (Choices | None): The choice matrix to be checked.

        Returns:
            bool: True if cm is None or contains a 0.
        """
        return cm is None or not np.all(cm)

    @staticmethod
    def accept(cm: Choices) -> np.bool:
        """Checks if all elements of the choice matrix are singletons.

        Assumes that cm does not contain a 0, which is true when this
        function is called in 'solution_generator'.

        Args:
            cm (Choices): The choice matrix to be checked.

        Returns:
            bool: True if all elements of cm are singletons.
        """
        return np.all(cm & (cm - 1) == 0)

    def expand(self, cm: Choices) -> list[Choices]:
        """Chooses a cell and lists the possible values for that cell.

        Expands the given choices matrix by selecting the element with
        the fewest possible choices, and creating new choice matrices
        for each possible choice of that element.

        This function may optionally be overridden by the user to make
        more informed guesses for the list of possible choice matrices.

        When overriding, you may make use of the following assumptions:
        - cm was not rejected => np.all(cm), i.e. no zeros in cm
        - cm was not accepted => there exists a cell of cm with c > 1

        When overriding, you must respect the following properties:
        If ems = expand(cm), then
        - Refinement: For all em in ems, em ⊊ cm
        - No solutions are lost: For all solutions xm ⊂ cm, there
          exists em in ems such that xm ⊂ em

        Args:
            cm (Choices): The choices matrix to expand.

        Returns:
            list[Choices]: A list of new choice matrices, each
                representing a possible choice for the element with the
                fewest possible choices.
        """
        powers_of_two = 1 << np.arange(32)
        multi_index = self.expand_cell(cm)
        powers_present = powers_of_two[cm[multi_index] & powers_of_two > 0]
        cm_copies = np.repeat(cm[np.newaxis, ...], len(powers_present), axis=0)
        cm_copies[:, *multi_index] = powers_present
        return list(cm_copies)

    def expand_cell(self, cm: Choices) -> tuple[np.intp, ...]:
        powers_of_two = 1 << np.arange(32)
        cardinality = np.sum((cm[..., None] & powers_of_two) != 0, axis=-1)
        cardinality_unfilled = np.where(cardinality == 1, np.inf, cardinality)
        multi_index = np.unravel_index(np.argmin(cardinality_unfilled), cm.shape)
        return multi_index

    def prune_repeatedly(self, cm: Choices) -> Choices | None:
        """Repeatedly calls prune until cm no longer changes.

        Args:
            cm (Choices): The choices to be pruned.

        Returns:
            Choices | None: The pruned choices, or None if the rules
                are violated.
        """
        prune_again = True
        while prune_again:
            cm_temp = np.copy(cm)
            cm_new = self.prune(cm)
            if self.reject(cm_new):
                return None
            cm = cm_new  # type: ignore[assignment]
            prune_again = not np.array_equal(cm, cm_temp)
        return cm

    def prune(self, cm: Choices) -> Choices | None:
        """Prunes the choices matrix based on the rules of the problem.

        Should be implemented by the user, since it is specific to the
        problem to be solved. Should obey the following properties:

        If om = prune(cm), then
        - Refinement: om ⊂ cm
        - No solutions are lost: xm ⊂ cm satisfies the rule => xm ⊂ om
        - Eventual rejection: If cm is all singletons and does not
          satisfy the rule, then reject(om) is True

        If cm will never lead to a valid solution, may just return None
        in the implementation.

        Args:
            cm (Choices): The input choices matrix.

        Returns:
            Choices | None: Pruned matrix or None
        """
        cm = np.copy(cm)
        for func in self.rules:
            cm = func(cm)
            if self.reject(cm):
                return None
        return cm

    def get_rules(self) -> list[Callable[[Choices], Choices | None]]:
        methods = inspect.getmembers(self.__class__, predicate=inspect.isfunction)
        rule_methods = [func for _, func in methods if getattr(func, "rule", False)]
        return rule_methods
