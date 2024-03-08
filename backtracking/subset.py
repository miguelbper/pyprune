from numba import njit
from typing import Optional
# TODO: write docs for all functions, describe domain.
# Include examples with subset creation (begin) & inspection (end).


def subset(xs: list[int]) -> int:
    return sum([1 << x for x in xs])


@njit
def elements(s: int) -> list[int]:
    x = 0
    result = []
    while s:
        if s & 1:
            result.append(x)
        s >>= 1
        x += 1
    return result


def smallest(s: int) -> Optional[int]:
    return None if is_empty(s) else (s & -s).bit_length() - 1


def num_elements(s: int) -> int:
    return s.bit_count()


def is_empty(s: int) -> bool:
    return not s


def is_singleton(s: int) -> bool:
    return s and s & (s - 1) == 0


def remove(s: int, x: int) -> int:
    return s & ~(1 << x)


def remove_except(s: int, x: int) -> int:
    return s & (1 << x)
