import numpy as np
from .utils import (
        dynamic_loss_plot_init,
        dynamic_loss_plot_update,
        dynamic_loss_plot_finish
    )
from tqdm import tqdm
from .layers import DeepNNLayer, SampleLayer
from .activations import ActivationFunc, Identity
from abc import ABC, abstractmethod

LOADER = ['⡿', '⣟', '⣯', '⣷', '⣾', '⣽', '⣻', '⢿']


class AAutoencoder(ABC):
    @abstractmethod
    def __init__(self,
                 encoder_layers: list[int],
                 decoder_layers: list[int],
                 lr: float,
                 activation_func: ActivationFunc):
        if encoder_layers[-1] != decoder_layers[0]:
            raise Exception(
                f"Encoder output and decoder input don't match {encoder_layers[-1]} != {encoder_layers[0]}" # noqa
            )
        self.encoder = DeepNNLayer(encoder_layers, lr, activation_func)
        self.decoder = DeepNNLayer(decoder_layers, lr, activation_func)
        self.space_dim = decoder_layers[0]
        self.lr = lr

    def train_dataset(self,
                      data_set: list[np.ndarray],
                      max_epoch: int,
                      patience: int,
                      display_loss: bool = False) -> list[float]:
        losses = [self.loss(data_set)]
        if display_loss is True:
            ax, line = dynamic_loss_plot_init(losses)
        epoch = 0
        no_improv = 0
        prev_error = losses[0]
        with tqdm(bar_format="{desc} {elapsed} {rate_fmt}") as lbar:
            while True:
                lbar.set_description(
                    f"{LOADER[epoch % len(LOADER)]} Training ({epoch=} error={float(prev_error):.6f})", # noqa
                )
                lbar.update()
                error = 0
                for x in tqdm(data_set, leave=False):
                    error += self.train(x)
                error /= len(data_set)
                derror = prev_error - error
                if derror <= 0 or abs(derror) < 1e-4:
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
            dynamic_loss_plot_finish()
        return losses

    def save(self, path: str):
        path = path.removesuffix('.npy')
        np.save(path, self)

    def load(path: str) -> 'ClassicalAutoencoder':
        path = path.removesuffix('.npy') + '.npy'
        data = np.load(path, allow_pickle=True)
        return data.item()

    @abstractmethod
    def loss(self, data_set: list[np.ndarray]) -> float:
        pass

    @abstractmethod
    def train(self, v: np.ndarray) -> float:
        pass

    @abstractmethod
    def forward(self, v: np.ndarray) -> np.ndarray:
        pass


class ClassicalAutoencoder(AAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f'Encoder:\n{self.encoder}\n\nDecoder:\n{self.decoder}'

    def loss(self, data_set: list[np.ndarray]) -> float:
        loss = 0
        for x in data_set:
            loss += np.sum(np.abs(x - self.forward(x)[0])) / len(x)
        return loss / len(data_set)

    def train(self, v: np.ndarray):
        out = self.decoder.forward(
            self.encoder.forward(v)
        )
        error = out - v
        self.encoder.backprop(
            self.decoder.backprop(error)
        )
        return np.sum(np.abs(error)) / len(v)

    def encode(self, v: np.ndarray) -> np.ndarray:
        return self.encoder.forward(v)

    def decode(self, v: np.ndarray) -> np.ndarray:
        return self.decoder.forward(v)

    def forward(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        code = self.encode(v)
        out = self.decode(code)
        return out, code


class VariationalAutoencoder(AAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = SampleLayer(self.encoder.out_size, self.lr, Identity())

    def loss(self, data_set: list[np.ndarray]) -> float:
        loss = 0
        for x in data_set:
            out = self.forward(x)[0]
            kl = self.sampler.DKL()
            loss += np.mean((out - x) ** 2)
            loss += kl
        return loss / len(data_set)

    def train(self, v: np.ndarray) -> float:
        out, _ = self.forward(v)
        error = out - v
        self.encoder.backprop(
            self.sampler.backprop(
                self.decoder.backprop(error)
            )
        )
        return np.mean(error ** 2) + self.sampler.DKL()

    def forward(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        code = self.encoder.forward(v)
        sample = self.sampler.forward(code)
        out = self.decoder.forward(sample)
        return out, code
