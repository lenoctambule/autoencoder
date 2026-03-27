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

    def backprop(self, error: np.ndarray) -> np.ndarray:
        dW = np.outer(self.last_input, error)
        self.W -= self.lr * dW
        self.B -= self.lr * error
        return error @ self.W.T


class DeepNNLayer:
    def __init__(self,
                 layers: list[int],
                 lr: float,
                 activation_func):
        self.layers: list[NNLayer] = []
        for i in range(len(layers) - 1):
            self.layers.append(
                    NNLayer(
                        layers[i],
                        layers[i+1],
                        lr,
                        activation_func)
                )

    def forward(self, v: np.ndarray) -> np.ndarray:
        v_i = v
        for layer in self.layers:
            v_i = layer.forward(v_i)
        return v_i

    def backprop(self, error: np.ndarray) -> np.ndarray:
        error_i = error
        for layer in self.layers[::-1]:
            error_i = layer.backprop(error_i)
        return error_i
