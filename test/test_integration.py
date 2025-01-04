from pyprune.backtracking import Backtracking, Choices


class NothingIsSolution(Backtracking):
    def __init__(self, cm: Choices) -> None:
        self.stack = [cm]

    @staticmethod
    def prune(cm: Choices) -> Choices | None:
        return None


class EverythingIsSolution(Backtracking):
    def __init__(self, cm: Choices) -> None:
        self.stack = [cm]

    @staticmethod
    def prune(cm: Choices) -> Choices | None:
        return cm


class OnlyZeros(Backtracking):
    def __init__(self, cm: Choices) -> None:
        self.stack = [cm]

    @staticmethod
    def prune(cm: Choices) -> Choices | None:
        return cm & (1 << 0)
