import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .autoencoder import AAutoencoder


class Plotter:
    def __init__(self, autoencoder: 'AAutoencoder'):
        pass

    def update(self):
        pass

    def close(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class CAPlotter(Plotter):
    def __init__(self, autoencoder: 'AAutoencoder'):
        self.autoencoder = autoencoder
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(
                list(range(len(autoencoder.losses))),
                autoencoder.losses,
                label="Loss"
            )
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training MSE Loss")
        self.ax.legend()
        self.update()

    def update(self):
        self.line.set_xdata(range(len(self.autoencoder.losses)))
        self.line.set_ydata(self.autoencoder.losses)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)

    def close(self):
        plt.ioff()
        plt.show()
