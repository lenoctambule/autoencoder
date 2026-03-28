import numpy as np
from utils import (dynamic_loss_plot_init,
                   dynamic_loss_plot_update,
                   dynamic_loss_plot_finish)
from tqdm import tqdm
from layers import DeepNNLayer

LOADER = ['⡿', '⣟', '⣯', '⣷', '⣾', '⣽', '⣻', '⢿']


class Autoencoder:
    def __init__(self,
                 encoder_layers: list[int],
                 decoder_layers: list[int],
                 lr: float,
                 activation_func):
        self.encoder = DeepNNLayer(encoder_layers, lr, activation_func)
        self.decoder = DeepNNLayer(decoder_layers, lr, activation_func)

    def train(self, v: np.ndarray):
        out = self.decoder.forward(
            self.encoder.forward(v)
        )
        self.encoder.backprop(
            self.decoder.backprop(out - v)
        )
        return np.sum(np.abs(out - v)) / len(v)

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
        print("Training complete !")
        if display_loss is True:
            dynamic_loss_plot_finish(ax, line)
        return losses

    def encode(self, v: np.ndarray) -> np.ndarray:
        return self.encoder.forward(v)

    def decode(self, v: np.ndarray) -> np.ndarray:
        return self.decoder.forward(v)
