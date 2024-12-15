import pytest

from pyprune.subset import (
    elements,
    is_empty,
    is_singleton,
    num_elements,
    remove,
    remove_except,
    smallest,
    subset,
)


@pytest.fixture(params=list(range(100)), ids=lambda x: f"[s={x}]")
def s(request) -> int:
    return request.param


class TestSubset:
    def test_subset_element_are_inverses(self, s: int):
        assert subset(elements(s)) == s

    def test_smallest(self):
        for j in range(10):
            for i in range(j):
                s = subset(list(range(i, j)))
                assert smallest(s) == i

    def test_num_elements(self, s: int):
        assert num_elements(s) == len(elements(s))

    def test_is_empty(self, s: int):
        assert s ^ is_empty(s)

    def test_is_singleton(self, s: int):
        assert (num_elements(s) == 1) == is_singleton(s)

    def test_remove(self, s: int):
        for x in range(max(elements(s), default=-1) + 1):
            r_sx_0 = remove(s, x)
            r_sx_1 = subset([y for y in elements(s) if y != x])
            assert r_sx_0 == r_sx_1

    def test_remove_except(self, s: int):
        for x in range(max(elements(s), default=-1) + 1):
            rex_sx_0 = remove_except(s, x)
            rex_sx_1 = subset([x] if x in elements(s) else [])
            assert rex_sx_0 == rex_sx_1
