import numpy as np
from enum import Enum
from typing import Callable

PointTransform: type = Callable[[int, int], tuple[int, int]]

class RotateMethod(Enum):
    R0 = "R0"

    @staticmethod
    def _r0(x, y):
        return x, y

    def apply(self, x, y):
        return RotateMethod._METHODS[self](x, y)

RotateMethod._METHODS = {
    RotateMethod.R0: RotateMethod._r0,
}