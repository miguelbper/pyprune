from timeit import timeit
from backtracking.subset import (
    subset,
    elements,
    smallest,
    num_elements,
    is_empty,
    is_singleton,
    remove,
    remove_except,
)


class TestSubset:
    def test_subset_element_are_inverses(self):
        for s in range(100):
            assert subset(elements(s)) == s

    def test_smallest(self):
        for j in range(10):
            for i in range(j):
                s = subset(list(range(i, j)))
                assert smallest(s) == i

    def test_num_elements(self):
        for s in range(100):
            assert num_elements(s) == len(elements(s))

    def test_is_empty(self):
        for s in range(100):
            assert s ^ is_empty(s)

    def test_is_singleton(self):
        for s in range(100):
            assert (num_elements(s) == 1) == is_singleton(s)

    def test_remove(self):
        for s in range(100):
            for x in range(max(elements(s), default=-1) + 1):
                r_sx_0 = remove(s, x)
                r_sx_1 = subset([y for y in elements(s) if y != x])
                assert r_sx_0 == r_sx_1

    def test_remove_except(self):
        for s in range(100):
            for x in range(max(elements(s), default=-1) + 1):
                rex_sx_0 = remove_except(s, x)
                rex_sx_1 = subset([x] if x in elements(s) else [])
                assert rex_sx_0 == rex_sx_1

    def test_numba_elements(self):
        fast_fn = elements
        slow_fn = fast_fn.py_func
        fast_fn(0)
        s = 133742069
        n = 10000
        t_fast = timeit(lambda: fast_fn(s), number=n)
        t_slow = timeit(lambda: slow_fn(s), number=n)
        assert t_fast < t_slow
