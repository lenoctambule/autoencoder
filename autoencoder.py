import numpy as np
from utils import regularize
import types


class Encoder:
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 lr: float,
                 activation_func: types.FunctionType):
        self.W = np.random.uniform(-1, 1, (in_size, out_size))
        self.lr = lr
        self.last_input = None
        self.last_output = None
        self.activation_func = activation_func

    def forward(self, V: np.ndarray) -> np.ndarray:
        self.last_input = V
        z = V @ self.W
        self.last_output = regularize(self.activation_func(z))
        return self.last_output

    def backprop(self, error: np.ndarray):
        dW = np.outer(self.last_input, error)
        self.W -= self.lr * dW
        return error @ self.W.T


class Decoder:
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 lr: float,
                 activation_func):
        self.W = np.random.uniform(-1, 1, (in_size, out_size))
        self.lr = lr
        self.last_input = None
        self.last_output = None
        self.activation_func = activation_func

    def forward(self, V: np.ndarray) -> np.ndarray:
        self.last_input = V
        z = V @ self.W
        self.last_output = regularize(self.activation_func(z))
        return self.last_output

    def backprop(self, target: np.ndarray):
        error = self.last_output - target
        dW = np.outer(self.last_input, error)
        self.W -= self.lr * dW
        return error @ self.W.T


class Autoencoder:
    def __init__(self,
                 in_len: int,
                 bottleneck: int,
                 lr: float,
                 activation_func):
        self.encoder = Encoder(in_len, bottleneck, lr, activation_func)
        self.decoder = Decoder(bottleneck, in_len, lr, activation_func)

    def train(self, v: np.ndarray) -> float:
        encoded = self.encoder.forward(v)
        reconstructed = self.decoder.forward(encoded)
        error = self.decoder.backprop(v)
        self.encoder.backprop(error)
        error = v - reconstructed
        return np.sum(np.abs(error))

    def encode(self, v: np.ndarray) -> np.ndarray:
        return self.encoder.forward(v)

    def decode(self, v: np.ndarray) -> np.ndarray:
        return self.decoder.forward(v)
