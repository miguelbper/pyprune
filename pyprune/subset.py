"""Provides functions for working with sets represented as integers.

An integer can be used to represent a set of elements: the binary
representation of the integer has 1s at the indices of the elements in
the set. For example, the integer 5 has binary representation 101, which
means that the set {0, 2} is represented by 5.

In the context of constraint satisfaction problems, in each cell of a
choices matrix we store a set of possible values, and the backtracking
algorithm progressively reduces the number of choices in each cell.
The operations necessary to do this are much faster if done with ints
representing sets plus bit operations, rather than using Python sets.
"""


def subset(xs: list[int]) -> int:
    """Converts a list of integers into a binary representation.

    Args:
        xs (list[int]): The list of integers.

    Returns:
        int: Number whose binary representation has 1s at the indices
            specified by the input list.
    """
    return sum([1 << x for x in xs])


def elements(s: int) -> list[int]:
    """Converts an integer into the set it represents.

    Args:
        s (int): The input integer.

    Returns:
        list[int]: A list of indices where the binary representation of
            the input integer has 1s.
    """
    x = 0
    result = []
    while s:
        if s & 1:
            result.append(x)
        s >>= 1
        x += 1
    return result


def smallest(s: int) -> int:
    """Finds the smallest element in the given set.

    Args:
        s (int): The set to find the smallest element from.

    Returns:
        int: The smallest element in the set, or -1 if the set is empty.
    """
    s = int(s)  # s may be a numpy integer, convert to Python int
    return -1 if is_empty(s) else (s & -s).bit_length() - 1


def num_elements(s: int) -> int:
    """Returns the number of elements in the set represented by 's'.

    Parameters:
        s (int): The input integer.

    Returns:
        int: The number of elements in 's'.
    """
    return int(s).bit_count()


def is_empty(s: int) -> bool:
    """Checks if the set represented by the given integer is empty.

    Args:
        s (int): The integer to check.

    Returns:
        bool: True if the set is empty, False otherwise.
    """
    return not s


def is_singleton(s: int) -> bool:
    """Checks if the given integer is a power of 2. This is equivalent to
    saying that the integer has exactly one bit set to 1.

    Args:
        s (int): The integer to be checked.

    Returns:
        bool: True if the integer is a power of 2, False otherwise.
    """
    return bool(s and s & (s - 1) == 0)


def remove(s: int, x: int) -> int:
    """Removes the bit at position x from the integer s.

    Parameters:
        s (int): The integer from which to remove the bit.
        x (int): The position of the bit to be removed.

    Returns:
        int: The updated integer with the bit at position x removed.
    """
    return s & ~(1 << x)


def remove_except(s: int, x: int) -> int:
    """Removes all bits from the integer `s` except for the bit at position
    `x`.

    Parameters:
        s (int): The input integer.
        x (int): The position of the bit to keep.

    Returns:
        int: The modified integer with all bits except for the bit at
            position `x` set to 0.
    """
    return s & (1 << x)
