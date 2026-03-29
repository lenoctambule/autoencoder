import numpy as np
from abc import ABC, abstractmethod


class ActivationFunc(ABC):
    @abstractmethod
    def derivative(v: np.ndarray) -> np.ndarray:
        pass


class ReLU(ActivationFunc):
    def __call__(self, x):
        return x * (x > 0)

    def derivative(self, x):
        return x > 0


class LeakyReLU(ActivationFunc):
    def __init__(self, k=0.01):
        self.k = k

    def __call__(self, x):
        return x * (x > 0) + self.k * x * (x <= 0)

    def derivative(self, x):
        return (x > 0) + self.k * (x <= 0)
