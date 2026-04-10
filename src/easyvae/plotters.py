import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .autoencoder import AAutoencoder, VariationalAutoencoder


class Plotter:
    def __init__(self, autoencoder: 'AAutoencoder'):
        pass

    def update(self):
        pass

    def close(self):
        pass

    def __del__(self):
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
        plt.close(self.fig)


class VAEPlotter(Plotter):
    def __init__(self, autoencoder: 'VariationalAutoencoder'):
        self.autoencoder = autoencoder
        plt.ion()
        self.fig, (self.ax_recon, self.ax_dkl) = plt.subplots(1, 2)
        self.line, = self.ax_recon.plot(
                list(range(len(self.autoencoder.recon_losses))),
                self.autoencoder.recon_losses,
                label="Loss"
            )
        self.ax_recon.set_xlabel("Epoch")
        self.ax_recon.set_ylabel("Loss")
        self.ax_recon.set_title("Reconstruction MSE Loss")
        self.ax_recon.legend()

        self.dkl_line, = self.ax_dkl.plot(
            list(range(len(self.autoencoder.KL_losses))),
            self.autoencoder.KL_losses,
            label="DKL Loss",
        )
        self.ax_dkl.set_xlabel("Epoch")
        self.ax_dkl.set_ylabel("Loss")
        self.ax_dkl.set_title("DKL Loss")
        self.ax_dkl.legend()
        self.update()

    def update(self):
        self.line.set_xdata(range(len(self.autoencoder.recon_losses)))
        self.line.set_ydata(self.autoencoder.recon_losses)
        self.ax_recon.relim()
        self.ax_recon.autoscale_view()

        self.dkl_line.set_xdata(range(len(self.autoencoder.KL_losses)))
        self.dkl_line.set_ydata(self.autoencoder.KL_losses)
        self.ax_dkl.relim()
        self.ax_dkl.autoscale_view()

        plt.draw()
        plt.pause(0.1)

    def close(self):
        plt.ioff()
        plt.close(self.fig)
