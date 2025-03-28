"""Backtracking algorithm for solving constraint satisfaction problems.

Offers a Backtracking class that implements a backtracking algorithm.
The class is general purpose and can be used to solve any constraint
satisfaction puzzle. However, the class does not know the rules of any
specific problem. Users should inherit from this class and add the rules
of the problem.
"""

import inspect
from collections.abc import Iterator
from copy import deepcopy
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

Int: TypeAlias = np.int32
BitMask: TypeAlias = np.int32
ArrayInt: TypeAlias = NDArray[Int]
ArrayBitMask: TypeAlias = NDArray[BitMask]
Rule: TypeAlias = Any  # Should be Callable[[ArrayBitMask], ArrayBitMask | None], using Any to avoid type checker issues

IS_RULE: str = "is_rule"


def rule(func: Rule) -> Rule:
    """Decorator to mark a method as a rule for the backtracking algorithm.

    Args:
        func (Rule): The function to be marked as a rule.

    Returns:
        Rule: The decorated function.
    """
    setattr(func, IS_RULE, True)  # noqa: B010
    return func


class Backtracking:
    """Represents a backtracking problem.

    Usage:
        1. Define a new class that inherits from this class.

        2. __init__:
            - Override
            - Do super().__init__()

        3. branch -> branch_cell
            Options (from less to more "manual")
            - Leave the methods as is / do nothing
            - Override branch_cell to specify what cell should be chosen
            - Override branch to specify different logic

        4. prune_repeatedly -> prune -> @rule's
            Options (from less to more "manual")
            - Define methods decorated with @rule, they will be called by prune
            - Override prune
            - Override prune_repeatedly

        5. Instantiate and call 'solution' or 'solutions' to find the solution(s).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes a Backtracking object.

        Args:
            Whatever you want to pass as argument.

        Returns:
            None
        """
        self.rules = self.get_rules()

    def solution_generator(self, stack: list[ArrayBitMask]) -> Iterator[ArrayInt]:
        """Generates solutions using backtracking algorithm.

        Generator that is called from the 'solution' and 'solutions'
        methods. A generator is used to prevent code duplication.

        Yields:
            ArrayInt: A valid solution grid.
        """
        stack = deepcopy(stack)

        while stack:
            bm_prev = stack.pop()
            bm = self.prune_repeatedly(bm_prev)

            if bm is None:
                continue
            if self.accept(bm):
                yield self.grid(bm)
            else:
                stack += self.branch(bm)

    def solution(self, stack: list[ArrayBitMask]) -> ArrayInt | None:
        """Finds a solution using a backtracking algorithm.

        Returns:
            ArrayInt | None: The solution grid if found, None otherwise.
        """
        return next(self.solution_generator(stack), None)

    def solutions(self, stack: list[ArrayBitMask]) -> list[ArrayInt]:
        """Returns a list of all possible solutions for the problem.

        Returns:
            A list of ArrayInt objects representing the possible solutions.
        """
        return list(self.solution_generator(stack))

    @staticmethod
    def grid(bm: ArrayBitMask) -> ArrayInt:
        """Convert from a choices matrix to a grid.

        Assumes that all elements of bm are singletons. When used in
        'solution_generator', this is true because of the 'accept'
        function.

        Args:
            bm (ArrayBitMask): The input choices matrix.

        Returns:
            ArrayInt: The resulting grid.
        """
        return np.log2(bm).astype(np.int32)

    @staticmethod
    def reject(bm: ArrayBitMask | None) -> bool:
        """Checks if the choice matrix is invalid.

        Args:
            bm (ArrayBitMask | None): The choice matrix to be checked.

        Returns:
            bool: True if bm is None or contains a 0.
        """
        return bm is None or not np.all(bm)

    @staticmethod
    def accept(bm: ArrayBitMask) -> np.bool:
        """Checks if all elements of the choice matrix are singletons.

        Assumes that bm does not contain a 0, which is true when this
        function is called in 'solution_generator'.

        Args:
            bm (ArrayBitMask): The choice matrix to be checked.

        Returns:
            bool: True if all elements of bm are singletons.
        """
        return np.all(bm & (bm - 1) == 0)

    def branch(self, bm: ArrayBitMask) -> list[ArrayBitMask]:
        """Chooses a cell and lists the possible values for that cell.

        Branches the given choices matrix by selecting the element with
        the fewest possible choices, and creating new choice matrices
        for each possible choice of that element.

        This function may optionally be overridden by the user to make
        more informed guesses for the list of possible choice matrices.

        When overriding, you may make use of the following assumptions:
        - bm was not rejected => np.all(bm), i.e. no zeros in bm
        - bm was not accepted => there exists a cell of bm with c > 1

        When overriding, you must respect the following properties:
        If ems = branch(bm), then
        - Refinement: For all em in ems, em ⊊ bm
        - No solutions are lost: For all solutions xm ⊂ bm, there
          exists em in ems such that xm ⊂ em

        Args:
            bm (ArrayBitMask): The choices matrix to branch.

        Returns:
            list[ArrayBitMask]: A list of new choice matrices, each
                representing a possible choice for the element with the
                fewest possible choices.
        """
        powers_of_two = 1 << np.arange(32)
        multi_index = self.branch_cell(bm)
        powers_present = powers_of_two[bm[multi_index] & powers_of_two > 0]
        bm_copies = np.repeat(bm[np.newaxis, ...], len(powers_present), axis=0)
        bm_copies[:, *multi_index] = powers_present
        return list(bm_copies)

    def branch_cell(self, bm: ArrayBitMask) -> tuple[np.intp, ...]:
        """Find the cell with the fewest possible choices.

        Args:
            bm (ArrayBitMask): The choices matrix to analyze.

        Returns:
            tuple[np.intp, ...]: The multi-dimensional index of the cell with
                fewest possible choices, excluding cells that are already
                determined (have only one choice).
        """
        powers_of_two = 1 << np.arange(32)
        cardinality = np.sum((bm[..., None] & powers_of_two) != 0, axis=-1)
        cardinality_unfilled = np.where(cardinality == 1, np.inf, cardinality)
        multi_index = np.unravel_index(np.argmin(cardinality_unfilled), bm.shape)
        return multi_index

    def prune_repeatedly(self, bm: ArrayBitMask) -> ArrayBitMask | None:
        """Repeatedly calls prune until bm no longer changes.

        Args:
            bm (ArrayBitMask): The choices to be pruned.

        Returns:
            ArrayBitMask | None: The pruned choices, or None if the rules
                are violated.
        """
        prune_again = True
        while prune_again:
            bm_temp = np.copy(bm)
            bm_new = self.prune(bm)
            if self.reject(bm_new):
                return None
            assert bm_new is not None
            bm = bm_new
            prune_again = not np.array_equal(bm, bm_temp)
        return bm

    def prune(self, bm: ArrayBitMask) -> ArrayBitMask | None:
        """Prunes the choices matrix based on the rules of the problem.

        Should be implemented by the user, since it is specific to the
        problem to be solved. Should obey the following properties:

        If om = prune(bm), then
        - Refinement: om ⊂ bm
        - No solutions are lost: xm ⊂ bm satisfies the rule => xm ⊂ om
        - Eventual rejection: If bm is all singletons and does not
          satisfy the rule, then reject(om) is True

        If bm will never lead to a valid solution, may just return None
        in the implementation.

        Args:
            bm (ArrayBitMask): The input choices matrix.

        Returns:
            ArrayBitMask | None: Pruned matrix or None
        """
        bm = np.copy(bm)
        for func in self.rules:
            bm_temp = func(bm)
            if bm_temp is None:
                return None
            bm = bm_temp
            if self.reject(bm):
                return None
        return bm

    def get_rules(self) -> list[Rule]:
        """Get all methods marked with the @rule decorator.

        Returns:
            list[Rule]: A list of bound methods that were decorated with @rule.
        """
        rules: list[Rule] = []
        for name, member in inspect.getmembers(self.__class__, predicate=inspect.isfunction):
            if getattr(member, IS_RULE, False):
                rules.append(getattr(self, name))
        return rules
