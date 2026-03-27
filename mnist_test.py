import matplotlib.pyplot as plt
import numpy as np
import keras
from autoencoder import Autoencoder
from utils import (relu,
                   dynamic_loss_plot_init,
                   dynamic_loss_plot_update,
                   dynamic_loss_plot_finish)


def mnist_embed(
        bottleneck: int,
        max_epoch: int,
        patience: int,
        ):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    in_len = x_train[0].flatten().shape[0]
    autoencoder = Autoencoder(in_len, bottleneck, 0.001, relu)
    ax, line = dynamic_loss_plot_init()
    no_improv = 0
    prev_error = float('inf')
    losses = []
    epoch = 0
    x_train = x_train[:]
    while True:
        error = 0
        for x in x_train:
            input = x.flatten() / 255
            error += autoencoder.train(input)
        error /= len(x_train)
        if error - prev_error <= 1e-8:
            no_improv += 1
        else:
            no_improv = 0
        prev_error = error
        losses.append(error)
        dynamic_loss_plot_update(ax, line, losses)
        if no_improv > patience:
            break
        if epoch > max_epoch:
            break
        epoch += 1
    print("Done!")
    dynamic_loss_plot_finish(ax, line)
    example: np.ndarray = x_test[np.random.randint(0, len(x_test))]
    code = autoencoder.encode(example.flatten() / 255)
    output = autoencoder.decode(code)
    plt.subplot(1, 2, 1)
    plt.matshow(example, fignum=False)
    plt.subplot(1, 2, 2)
    plt.matshow(output.reshape(example.shape), fignum=False)
    plt.show()


if __name__ == "__main__":
    import argparse
    import sys

    options = "b:e:p:"
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, nargs='+', default=50)
    parser.add_argument('-e', type=int, nargs='+', default=1000)
    parser.add_argument('-p', type=int, nargs='+', default=5)
    args = parser.parse_args(sys.argv[1:])

    mnist_embed(args.b, args.e, args.p)
