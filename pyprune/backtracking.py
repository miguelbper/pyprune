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
from math import prod
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Int: TypeAlias = np.int32
BitMask: TypeAlias = np.uint32
ArrayInt: TypeAlias = NDArray[Int]
ArrayBitMask: TypeAlias = NDArray[BitMask]


def num_elements(bm: ArrayBitMask) -> int:
    return prod(x.bit_count() for x in bm.flat)


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
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initializes a Backtracking object.

        Args:
            Whatever you want to pass as argument.

        Returns:
            None
        """
        self.rules = self.get_rules()

    def solution_generator(self, stack: list[ArrayBitMask], verbose: bool = False) -> Iterator[ArrayInt]:
        """Generates solutions using backtracking algorithm.

        Generator that is called from the 'solution' and 'solutions'
        methods. A generator is used to prevent code duplication.

        Yields:
            ArrayInt: A valid solution grid.
        """
        stack = deepcopy(stack)

        self.track_progress = verbose and len(stack) == 1
        if self.track_progress:
            message_top = "Number of grids: "
            message_bot = "Seen or rejected: "
            num_message_chars = max(len(message_top), len(message_bot))
            bm0 = stack[0]
            self.num_total = num_elements(bm0)
            self.num_pruned = 0
            self.message_top_fmt = message_top.ljust(num_message_chars)
            self.message_bot_fmt = message_bot.ljust(num_message_chars)
            self.format_num = lambda x: str(x).rjust(len(str(self.num_total)))
            print(self.message_top_fmt, self.num_total)

        while stack:
            bm_prev = stack.pop()
            bm = self.prune_repeatedly(bm_prev)

            if self.track_progress:
                bm_curr = np.zeros_like(bm_prev) if bm is None else bm
                num_rejected = num_elements(bm_prev) - num_elements(bm_curr)
                self.num_pruned += num_rejected
                percentage = f"{100 * self.num_pruned / self.num_total:.4f}%"
                print(self.message_bot_fmt, f"{self.format_num(self.num_pruned)} => {percentage}", end="\r")

            if bm is None:
                continue
            if self.accept(bm):
                if self.track_progress:
                    num_rejected = 1
                    self.num_pruned += num_rejected
                    percentage = f"{100 * self.num_pruned / self.num_total:.4f}%"
                    print(self.message_bot_fmt, f"{self.format_num(self.num_pruned)} => {percentage}", end="\r")

                yield self.grid(bm)
            else:
                stack += self.expand(bm)

    def solution(self, stack: list[ArrayBitMask], verbose: bool = False) -> ArrayInt | None:
        """Finds a solution using a backtracking algorithm.

        Returns:
            ArrayInt | None: The solution grid if found, None otherwise.
        """
        ans = next(self.solution_generator(stack, verbose), None)
        if self.track_progress:
            print()  # Here to prevent overriting of the last progress print
        return ans

    def solutions(self, stack: list[ArrayBitMask], verbose: bool = False) -> list[ArrayInt]:
        """Returns a list of all possible solutions for the problem.

        Returns:
            A list of ArrayInt objects representing the possible solutions.
        """
        ans = list(self.solution_generator(stack, verbose))
        if self.track_progress:
            print()  # Here to prevent overriting of the last progress print
        return ans

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

    def expand(self, bm: ArrayBitMask) -> list[ArrayBitMask]:
        """Chooses a cell and lists the possible values for that cell.

        Expands the given choices matrix by selecting the element with
        the fewest possible choices, and creating new choice matrices
        for each possible choice of that element.

        This function may optionally be overridden by the user to make
        more informed guesses for the list of possible choice matrices.

        When overriding, you may make use of the following assumptions:
        - bm was not rejected => np.all(bm), i.e. no zeros in bm
        - bm was not accepted => there exists a cell of bm with c > 1

        When overriding, you must respect the following properties:
        If ems = expand(bm), then
        - Refinement: For all em in ems, em ⊊ bm
        - No solutions are lost: For all solutions xm ⊂ bm, there
          exists em in ems such that xm ⊂ em

        Args:
            bm (ArrayBitMask): The choices matrix to expand.

        Returns:
            list[ArrayBitMask]: A list of new choice matrices, each
                representing a possible choice for the element with the
                fewest possible choices.
        """
        powers_of_two = 1 << np.arange(32)
        multi_index = self.expand_cell(bm)
        powers_present = powers_of_two[bm[multi_index] & powers_of_two > 0]
        bm_copies = np.repeat(bm[np.newaxis, ...], len(powers_present), axis=0)
        bm_copies[:, *multi_index] = powers_present
        return list(bm_copies)

    def expand_cell(self, bm: ArrayBitMask) -> tuple[np.intp, ...]:
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
            bm = bm_new  # type: ignore[assignment]
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
            bm = func(bm)
            if self.reject(bm):
                return None
        return bm

    def get_rules(self) -> list[Callable[[ArrayBitMask], ArrayBitMask | None]]:
        rules = []
        for name, member in inspect.getmembers_static(self.__class__):
            is_static = isinstance(member, staticmethod)
            if not (inspect.isfunction(member) or is_static):
                continue
            func = member.__func__ if is_static else member
            if not (getattr(func, "rule", False) or getattr(member, "rule", False)):
                continue
            rules.append(getattr(self, name))
        return rules

    def optimize(
        self,
        stack: list[ArrayBitMask],
        maximize: bool,
        verbose: bool = False,
    ) -> tuple[ArrayInt, Int] | tuple[None, float]:
        sign = 1 if maximize else -1
        best_xm = None
        best_score: float = -sign * np.inf  # Start with worse possible score and improve from there

        stack = deepcopy(stack)
        while stack:
            bm = self.prune_repeatedly(stack.pop())
            if bm is None:
                continue

            # If current best score is better than all scores we could see, reject
            score = self.criterion(bm, best_score)
            if score is None or sign * (best_score - score) >= 0:
                continue

            if self.accept(bm):
                best_xm: ArrayInt = self.grid(bm)
                best_score: Int = score

                if verbose:
                    print(f"\n{best_score = }")
                    print("best_xm = \n", best_xm, sep="")
            else:
                stack += self.expand(bm)

        return best_xm, best_score

    def criterion(self, bm: ArrayBitMask, best_score: Int | float) -> Int | None:
        raise NotImplementedError("For optimization, criterion needs to be defined.")
