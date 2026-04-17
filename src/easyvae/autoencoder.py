import numpy as np
from tqdm import tqdm
from .layers import DeepNNLayer, SampleLayer, NoiseLayer
from .activations import ActivationFunc, Identity
from .plotters import Plotter, CAPlotter, VAEPlotter
from .utils import interruptable
from abc import ABC, abstractmethod

LOADER = ['⡿', '⣟', '⣯', '⣷', '⣾', '⣽', '⣻', '⢿']
SQRT_2PI = np.sqrt(2 * np.pi)


class AAutoencoder(ABC):
    plotter_cls = Plotter

    @abstractmethod
    def __init__(self,
                 encoder_layers: list[int],
                 decoder_layers: list[int],
                 lr: float,
                 activation_func: ActivationFunc,
                 noise=0):
        if encoder_layers[-1] != decoder_layers[0]:
            raise Exception(
                f"Encoder output and decoder input don't match {encoder_layers[-1]} != {encoder_layers[0]}" # noqa
            )
        self.encoder = DeepNNLayer(encoder_layers, lr, activation_func)
        self.decoder = DeepNNLayer(decoder_layers, lr, activation_func)
        self.noise = NoiseLayer(noise)
        self.space_dim = decoder_layers[0]
        self.lr = lr
        self.losses = [0]

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

    @abstractmethod
    def train_dataset(self, *args, **kwargs) -> list[float]:
        pass


class ClassicalAutoencoder(AAutoencoder):
    plotter_cls = CAPlotter

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses = []

    def __str__(self):
        return "\n".join((
                    f"Type: {__class__.__name__}",
                    "Encoder:",
                    f"{self.encoder}",
                    "Decoder:",
                    f"{self.decoder}"
                )
            )

    def loss(self, data_set: list[np.ndarray]) -> float:
        loss = 0
        for x in data_set:
            loss += np.sum(np.abs(x - self.forward(x)[0])) / len(x)
        return loss / len(data_set)

    def train(self, v: np.ndarray):
        out, _ = self.forward(
            self.noise.forward(v)
        )
        error = out - v
        self.encoder.back(
            self.decoder.back(error)
        )
        self.encoder.backprop()
        self.decoder.backprop()
        return np.sum(np.abs(error)) / len(v)

    @interruptable
    def train_dataset(self,
                      data_set: list[np.ndarray],
                      max_epoch: int,
                      patience: int,
                      display_loss: bool = False) -> list[float]:
        plotter = self.plotter_cls(self) if display_loss else Plotter(self)
        if len(self.losses) == 0:
            self.losses = [self.loss(data_set)]
        epoch = 0
        no_improv = 0
        prev_error = self.losses[0]
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
                if abs(derror) < 1e-4:
                    no_improv += 1
                else:
                    no_improv = 0
                prev_error = float(error)
                self.losses.append(error)
                if no_improv > patience:
                    break
                if epoch > max_epoch:
                    break
                plotter.update()
                epoch += 1

    def encode(self, v: np.ndarray) -> np.ndarray:
        return self.encoder.forward(v)

    def decode(self, v: np.ndarray) -> np.ndarray:
        return self.decoder.forward(v)

    def forward(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        code = self.encode(v)
        out = self.decode(code)
        return out, code


class VariationalAutoencoder(AAutoencoder):
    plotter_cls = VAEPlotter

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = SampleLayer(self.encoder.out_size, self.lr, Identity())
        self.KL_losses = []
        self.recon_losses = []

    def __str__(self):
        return "\n".join((
                f"Type: {__class__.__name__}",
                "Encoder:",
                f"{self.encoder}",
                "Decoder:",
                f"{self.decoder}"
            ))

    def loss(self, data_set: list[np.ndarray]) -> float:
        kl_loss = 0
        recon_loss = 0
        for x in data_set:
            out = self.forward(x)[0]
            kl = self.sampler.DKL()
            recon_loss += np.mean((out - x) ** 2)
            kl_loss += kl
        kl_loss /= len(data_set)
        recon_loss /= len(data_set)
        return recon_loss, kl_loss

    def train(self, v: np.ndarray) -> tuple[float, float]:
        out, _ = self.forward(
            self.noise.forward(v)
        )
        error = out - v
        self.encoder.back(
            self.sampler.back(
                self.decoder.back(error)
            )
        )
        self.encoder.backprop()
        self.sampler.backprop()
        self.decoder.backprop()
        return np.mean(error ** 2), self.sampler.DKL()

    @interruptable
    def train_dataset(self,
                      data_set: list[np.ndarray],
                      max_epoch: int,
                      patience: int,
                      display_loss: bool = False) -> list[float]:
        plotter = self.plotter_cls(self) if display_loss else Plotter(self)
        if len(self.recon_losses) == 0:
            recon_0, kl_0 = self.loss(data_set)
            self.recon_losses = [recon_0]
            self.KL_losses = [kl_0]
        epoch = 0
        no_improv = 0
        prev_loss = self.recon_losses[0] + self.KL_losses[0]
        with tqdm(bar_format="{desc} {elapsed} {rate_fmt}") as lbar:
            while True:
                lbar.set_description(
                    f"{LOADER[epoch % len(LOADER)]} Training ({epoch=} loss={float(prev_loss):.6f})", # noqa
                )
                lbar.update()
                dkl = 0
                recon = 0
                for x in tqdm(data_set, leave=False):
                    recon_i, dkl_i = self.train(x)
                    dkl += dkl_i
                    recon += recon_i
                recon /= len(data_set)
                dkl /= len(data_set)
                loss = recon + dkl
                dloss = prev_loss - loss
                if dloss <= 0 or abs(dloss) < 1e-4:
                    no_improv += 1
                else:
                    no_improv = 0
                prev_loss = float(loss)
                self.recon_losses.append(recon)
                self.KL_losses.append(dkl)
                if no_improv > patience:
                    break
                if epoch > max_epoch:
                    break
                plotter.update()
                epoch += 1

    def forward(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        code = self.encoder.forward(v)
        sample = self.sampler.forward(code)
        out = self.decoder.forward(sample)
        return out, sample

    def encode(self, v: np.ndarray) -> np.ndarray:
        return self.sampler.forward(
            self.encoder.forward(v)
        )

    def decode(self, v: np.ndarray) -> np.ndarray:
        return self.decoder.forward(v)


class Label:
    def __init__(self,
                 name: str,
                 embedding_size: int,
                 N=100):
        self.name = name
        self.embedding_size = embedding_size
        self.N = N
        self.idx = 0
        self.history = np.zeros((self.N, embedding_size))

    def observe(self, code: np.ndarray):
        if self.idx < self.N:
            self.history[self.idx] = code
            self.idx += 1
        else:
            diffs = np.linalg.norm(self.history - code, axis=1)
            idx = np.argmin(diffs)
            self.history[idx] = (self.history[idx] + code) / 2

    def p(self, x: np.ndarray):
        return np.mean(
            np.exp(-np.abs(self.history - x))
        )


class LabelingVAE(VariationalAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels: list[Label] = []
        self.labels_idxs: dict[str, int] = {}

    def learn_labels(self, data: np.ndarray, labels: list[list[str]]):
        self.labels.clear()
        self.labels_idxs.clear()
        for x_i, labels_i in zip(data, labels):
            y_i = self.encode(x_i)
            for c in labels_i:
                idx = self.labels_idxs.get(c, None)
                if idx is None:
                    label = Label(c, self.encoder.out_size)
                    self.labels.append(label)
                    self.labels_idxs[c] = len(self.labels) - 1
                else:
                    label = self.labels[idx]
                label.observe(y_i)

    def label(self, x: np.ndarray):
        probs = {}
        total = 0
        code = self.encode(x)
        for label in self.labels:
            p = label.p(code)
            probs[label.name] = p
            total += p
        for k in probs:
            probs[k] = float(probs[k] / total)
        return dict(
            sorted(
                probs.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )
