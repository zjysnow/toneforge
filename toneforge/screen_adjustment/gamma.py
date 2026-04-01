import numpy as np
from scipy.optimize import curve_fit
from abc import ABC, abstractmethod

class GammaModel(ABC):
    def __init__(self):
        self.alpha = 0
        self.gamma = 0

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.alpha * (x ** self.gamma)
    
    
class LogTwoPoint(GammaModel):
    def __init__(self):
        super().__init__()

    def fit(self, x: np.ndarray, y: np.ndarray):
        if x.size != 2 or y.size !=2:
            raise ValueError("x and y should be 1d arrays of size 2")

        T1, T2 = np.maximum(y, 1e-6)
        C1, C2 = np.maximum(x, 1e-6)

        self.gamma = np.log(T1/T2) / np.log(C1/C2)
        self.alpha = T1 / (C1**self.gamma)

    
class CurveFit(GammaModel):
    def __init__(self):
        super().__init__()

    def fit(self, x: np.ndarray, y: np.ndarray):
        def fn(x, alpha, gamma):
            return np.clip(alpha * (x ** gamma), 0, 1)
        (self.alpha, self.gamma), _ = curve_fit(fn, x, y, p0=[1.0, 2.2])

    