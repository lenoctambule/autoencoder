import numpy as np
import types
from utils import normalize


class NNLayer:
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 lr: float,
                 activation_func: types.FunctionType):
        self.W = np.random.uniform(-1, 1, (in_size, out_size))
        self.B = np.zeros((out_size))
        self.lr = lr
        self.input = None
        self.output = None
        self.output_linear = None
        self.activation_func = activation_func

    def __str__(self):
        return f'[ {self.W.shape[0]} => {self.W.shape[1]}\tlr:{self.lr}\tactivation:{self.activation_func.__name__} ]' # noqa

    def forward(self, V: np.ndarray) -> np.ndarray:
        self.input = normalize(V)
        self.output_linear = self.input @ self.W + self.B
        self.output = self.activation_func(
                self.output_linear
            )
        return self.output

    def backprop(self, error: np.ndarray) -> np.ndarray:
        error *= self.activation_func(self.output_linear, True)
        ret = self.W @ error
        dW = np.outer(self.input, error) * self.lr
        dB = error * self.lr
        self.W -= dW
        self.B -= dB
        return ret


class DeepNNLayer:
    def __init__(self,
                 layers: list[int],
                 lr: float,
                 activation_func: types.FunctionType):
        self.layers: list[NNLayer] = []
        for i in range(len(layers) - 1):
            self.layers.append(
                    NNLayer(
                        layers[i],
                        layers[i+1],
                        lr,
                        activation_func)
                )

    def __str__(self):
        return '\n'.join([str(layer) for layer in self.layers])

    def forward(self, v: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            v = layer.forward(v)
        return v

    def backprop(self, error: np.ndarray) -> np.ndarray:
        for layer in self.layers[::-1]:
            error = layer.backprop(error)
        return error
