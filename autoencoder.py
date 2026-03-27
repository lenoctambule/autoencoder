import numpy as np
from utils import (dynamic_loss_plot_init,
                   dynamic_loss_plot_update,
                   dynamic_loss_plot_finish)
from tqdm import tqdm
from layers import NNLayer

LOADER = ['⡿', '⣟', '⣯', '⣷', '⣾', '⣽', '⣻', '⢿']


class Autoencoder:
    def __init__(self,
                 in_len: int,
                 bottleneck: int,
                 lr: float,
                 activation_func):
        self.encoder = NNLayer(in_len, bottleneck, lr, activation_func)
        self.decoder = NNLayer(bottleneck, in_len, lr, activation_func)

    def train(self, v: np.ndarray) -> float:
        encoded = self.encoder.forward(v)
        reconstructed = self.decoder.forward(encoded)
        error = self.decoder.backprop(reconstructed - v)
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
        with tqdm(bar_format="{desc} {elapsed} {rate_fmt}") as lbar:
            while True:
                lbar.set_description(
                    f"{LOADER[epoch % len(LOADER)]} Training ({epoch=} error={prev_error:.2f})", # noqa
                )
                lbar.update()
                error = 0
                for x in data_set:
                    input = x.flatten()
                    error += self.train(input)
                error /= len(data_set)
                if prev_error - error <= 1e-8:
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
            print("#Training complete !")
            return losses

    def encode(self, v: np.ndarray) -> np.ndarray:
        return self.encoder.forward(v)

    def decode(self, v: np.ndarray) -> np.ndarray:
        return self.decoder.forward(v)
