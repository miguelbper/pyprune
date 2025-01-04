from pyprune.backtracking import ArrayBitMask, Backtracking


class NothingIsSolution(Backtracking):
    def __init__(self, cm: ArrayBitMask) -> None:
        self.stack = [cm]

    @staticmethod
    def prune(cm: ArrayBitMask) -> ArrayBitMask | None:
        return None


class EverythingIsSolution(Backtracking):
    def __init__(self, cm: ArrayBitMask) -> None:
        self.stack = [cm]

    @staticmethod
    def prune(cm: ArrayBitMask) -> ArrayBitMask | None:
        return cm


class OnlyZeros(Backtracking):
    def __init__(self, cm: ArrayBitMask) -> None:
        self.stack = [cm]

    @staticmethod
    def prune(cm: ArrayBitMask) -> ArrayBitMask | None:
        return cm & (1 << 0)
