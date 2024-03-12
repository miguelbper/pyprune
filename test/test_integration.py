import numpy as np
from typing import Optional
from backtracking.subset import remove_except, num_elements
from backtracking.backtracking import Backtracking, Choices
from itertools import product


rng = np.random.default_rng(1337)


class NothingIsSolution(Backtracking):
    def __init__(self, cm0: Choices) -> None:
        super().__init__(cm0)

    def constraint_nothing_is_solution(self, cm: Choices) -> Optional[Choices]:
        return None


class EverythingIsSolution(Backtracking):
    def __init__(self, cm0: Choices) -> None:
        super().__init__(cm0)


class OnlyZeros(Backtracking):
    def __init__(self, cm0: Choices) -> None:
        super().__init__(cm0)

    def constraint_only_zeros(self, cm: Choices) -> Optional[Choices]:
        m, n = cm.shape
        ans = np.copy(cm)
        for i, j in product(range(m), range(n)):
            ans[i, j] = remove_except(cm[i, j], 0)
        return ans


class TestIntegration:
    def test_nothing_is_solution(self):
        for _ in range(10):
            cm = rng.integers(1, 2**4, (5, 5), dtype=np.uint32)
            problem = NothingIsSolution(cm)
            assert problem.solution() is None
            assert problem.solutions() == []

    def test_everything_is_solution(self):
        for _ in range(5):
            cm = rng.integers(1, 2**3, (2, 2), dtype=np.uint32)
            problem = EverythingIsSolution(cm)
            num_choices = np.vectorize(num_elements)(cm)
            assert len(problem.solutions()) == np.prod(num_choices)

    def test_only_zeros(self):
        for _ in range(10):
            cm = rng.integers(1, 2**4, (5, 5), dtype=np.uint32)
            zeros = np.zeros((5, 5), dtype=np.uint32)
            ones = np.ones((5, 5), dtype=np.uint32)
            cm += np.where(cm % 2, zeros, ones)
            problem = OnlyZeros(cm)
            assert np.array_equal(problem.solution(), zeros)
            assert len(problem.solutions()) == 1
