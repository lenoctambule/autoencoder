import numpy as np
from utils import (regularize,
                   dynamic_loss_plot_init,
                   dynamic_loss_plot_update,
                   dynamic_loss_plot_finish)
import types

LOADER = ['⡿', '⣟', '⣯', '⣷', '⣾', '⣽', '⣻', '⢿']


class Encoder:
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


class Decoder:
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 lr: float,
                 activation_func):
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

    def backprop(self, target: np.ndarray):
        error = self.last_output - target
        dW = np.outer(self.last_input, error)
        self.W -= self.lr * dW
        self.B -= self.lr * error
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

    def train_dataset(self,
                      data_set: list[np.ndarray],
                      max_epoch: int,
                      patience: int,
                      display_loss: bool = False) -> list[float]:
        if display_loss is True:
            ax, line = dynamic_loss_plot_init()
        losses = []
        epoch = 0
        no_improv = 0
        prev_error = float('inf')
        while True:
            print(
                f"{LOADER[epoch % len(LOADER)]} Training \t({epoch=} error={prev_error:.2f})", # noqa
                end="\r"
            )
            error = 0
            for x in data_set:
                input = x.flatten()
                error += self.train(input)
            error /= len(data_set)
            if error - prev_error <= 1e-8:
                no_improv += 1
            else:
                no_improv = 0
            prev_error = float(error)
            losses.append(error)
            if display_loss is True:
                dynamic_loss_plot_update(ax, line, losses)
            if no_improv > patience:
                break
            if epoch > max_epoch:
                break
            epoch += 1
        if display_loss is True:
            dynamic_loss_plot_finish(ax, line)
        return losses

    def encode(self, v: np.ndarray) -> np.ndarray:
        return self.encoder.forward(v)

    def decode(self, v: np.ndarray) -> np.ndarray:
        return self.decoder.forward(v)
