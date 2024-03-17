import pytest
from backtracking.subset import (
    subset,
    elements,
    smallest,
    smallest_numba,
    num_elements,
    num_elements_numba,
    is_empty,
    is_singleton,
    is_singleton_numba,
    remove,
    remove_except,
)


@pytest.fixture(params=list(range(100)), ids=lambda x: f"[s={x}]")
def s(request):
    return request.param


class TestSubset:
    def test_subset_element_are_inverses(self, s):
        assert subset(elements(s)) == s

    def test_smallest(self):
        for j in range(10):
            for i in range(j):
                s = subset(list(range(i, j)))
                assert smallest(s) == i

    def test_num_elements(self, s):
        assert num_elements(s) == len(elements(s))

    def test_is_empty(self, s):
        assert s ^ is_empty(s)

    def test_is_singleton(self, s):
        assert (num_elements(s) == 1) == is_singleton(s)

    def test_remove(self, s):
        for x in range(max(elements(s), default=-1) + 1):
            r_sx_0 = remove(s, x)
            r_sx_1 = subset([y for y in elements(s) if y != x])
            assert r_sx_0 == r_sx_1

    def test_remove_except(self, s):
        for x in range(max(elements(s), default=-1) + 1):
            rex_sx_0 = remove_except(s, x)
            rex_sx_1 = subset([x] if x in elements(s) else [])
            assert rex_sx_0 == rex_sx_1


class TestSubsetNumba:
    def test_smallest_numba(self, s):
        assert smallest(s) == smallest_numba(s)

    def test_num_elements_numba(self, s):
        assert num_elements(s) == num_elements_numba(s)

    def test_is_singleton_numba(self, s):
        assert is_singleton(s) == is_singleton_numba(s)
