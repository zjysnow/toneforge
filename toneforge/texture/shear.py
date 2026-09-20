import numpy as np
from enum import Enum
from typing import Callable

PointTransform: type = Callable[[int, int], tuple[int, int]]

class ShearMethod(Enum):
    X0 = "X0"
    X8 = "X8"
    X18 = "X18"
    X31 = "X31"
    X45 = "X45"
    Y8 = "Y8"
    Y18 = "Y18"
    Y31 = "Y31"
    Y45 = "Y45"

    def apply(self, x, y):
        return ShearMethod._METHODS[self](x, y)

    @staticmethod
    def _x0(x, y): 
        return x, y

    @staticmethod
    def _x8(x, y):
        return x-(x>>3)+(y>>3), y

    @staticmethod
    def _x18(x, y):
        return x-(x>>2)+(y>>2), y

    @staticmethod
    def _x31(x, y):
        return (x>>1)+(x>>3)+(y>>2)+(y>>3), y
    
    @staticmethod
    def _x45(x, y):
        return (x>>1)+(y>>1), y

    @staticmethod
    def _y8(x, y):
        return x, y-(y>>3)+(x>>3)

    @staticmethod
    def _y18(x, y):
        return x, y-(y>>2)+(x>>2)

    @staticmethod
    def _y31(x, y):
        return x, (y>>1)+(y>>3)+(x>>2)+(x>>3)

    @staticmethod
    def _y45(x, y):
        return x, (x>>1)+(y>>1)

ShearMethod._METHODS = {
    ShearMethod.X0: ShearMethod._x0,
    ShearMethod.X8: ShearMethod._x8,
    ShearMethod.X18: ShearMethod._x18,
    ShearMethod.X31: ShearMethod._x31,
    ShearMethod.X45: ShearMethod._x45,
    ShearMethod.Y8: ShearMethod._y8,
    ShearMethod.Y18: ShearMethod._y18,
    ShearMethod.Y31: ShearMethod._y31,
    ShearMethod.Y45: ShearMethod._y45,
}