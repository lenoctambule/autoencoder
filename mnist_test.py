import matplotlib.pyplot as plt
import numpy as np
from autoencoder import Autoencoder
from utils import relu


def load_mnist():
    import os
    import requests

    mnist_path = "./mnist.npz"
    mnist_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    if not os.path.exists(mnist_path):
        with open(mnist_path, "w+b") as f:
            f.write(requests.get(mnist_url, stream=True).content)
    res = np.load(mnist_path)
    return res["x_train"], res["y_train"], res["x_test"], res["y_test"]


def mnist_test(
        bottleneck: int,
        max_epoch: int,
        patience: int,
        ):
    x_train, _, x_test, _ = load_mnist()
    x_train = np.divide(x_train, 255)
    x_test = np.divide(x_train, 255)
    in_len = x_train[0].flatten().shape[0]
    autoencoder = Autoencoder(in_len, bottleneck, 0.001, relu)
    autoencoder.train_dataset(x_train, max_epoch, patience, display_loss=True)
    example: np.ndarray = x_test[np.random.randint(0, len(x_test))]
    code = autoencoder.encode(example.flatten())
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
    parser.add_argument('-b', type=int, nargs='?', default=50)
    parser.add_argument('-e', type=int, nargs='?', default=1000)
    parser.add_argument('-p', type=int, nargs='?', default=5)
    args = parser.parse_args(sys.argv[1:])
    mnist_test(args.b, args.e, args.p)
