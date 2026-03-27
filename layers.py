import numpy as np
import types
from utils import regularize


class NNLayer:
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 lr: float,
                 activation_func: types.FunctionType):
        self.W = np.random.uniform(-1, 1, (in_size, out_size))
        self.B = np.zeros((out_size))
        self.lr = lr
        self.last_input = None
        self.last_output = None
        self.activation_func = activation_func

    def forward(self, V: np.ndarray) -> np.ndarray:
        self.last_input = V
        res = V @ self.W + self.B
        self.last_output = regularize(self.activation_func(res))
        return self.last_output

    def backprop(self, error: np.ndarray):
        dW = np.outer(self.last_input, error)
        self.W -= self.lr * dW
        self.B -= self.lr * error
        return error @ self.W.T
