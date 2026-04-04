import numpy as np
from utils import normalize
from activations import ActivationFunc


class NNLayer:
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 lr: float,
                 activation_func: ActivationFunc):
        self.W = np.random.uniform(-1, 1, (in_size, out_size))
        self.B = np.zeros((out_size))
        self.lr = lr
        self.input = None
        self.output = None
        self.output_linear = None
        self.activation_func = activation_func

    def __str__(self):
        return f'[ {self.W.shape[0]} => {self.W.shape[1]}\tlr:{self.lr}\tactivation:{self.activation_func.__class__.__name__} ]' # noqa

    def forward(self, v: np.ndarray) -> np.ndarray:
        self.input = normalize(v)
        self.output_linear = self.input @ self.W + self.B
        self.output = self.activation_func(
                self.output_linear
            )
        return self.output

    def backprop(self, error: np.ndarray) -> np.ndarray:
        error *= self.activation_func.d(self.output_linear)
        ret = self.W @ error
        dW = np.outer(self.input, error) * self.lr
        dB = error * self.lr
        self.W -= dW
        self.B -= dB
        return ret


class SampleLayer:
    def __init__(self,
                 in_size: int,
                 lr: float,
                 activation_func: ActivationFunc):
        self.input = None
        self.mean_nn = NNLayer(
            in_size,
            in_size,
            lr,
            activation_func)
        self.std_nn = NNLayer(
            in_size,
            in_size,
            lr,
            activation_func)

    def forward(self, v: np.ndarray) -> np.ndarray:
        self.input = v
        self.mean = self.mean_nn.forward(v)
        self.std = self.std_nn.forward(v)
        self.eps = np.random.normal(0, 1)
        return self.eps * self.std + self.mean

    def backprop(self, error: np.ndarray) -> np.ndarray:
        mu_error = self.mean_nn.backprop(error)
        std_error = self.std_nn.backprop(self.eps * error)
        return mu_error + std_error


class DeepNNLayer:
    def __init__(self,
                 layers: list[int],
                 lr: float,
                 activation_func: ActivationFunc):
        self.layers: list[NNLayer] = []
        for i in range(len(layers) - 1):
            self.layers.append(
                    NNLayer(
                        layers[i],
                        layers[i+1],
                        lr,
                        activation_func)
                )
        self.in_size = layers[0]
        self.out_size = layers[-1]

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
