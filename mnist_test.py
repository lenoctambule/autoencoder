import matplotlib.pyplot as plt
import numpy as np
from autoencoder import Autoencoder
from utils import relu


def load_mnist() -> list[np.ndarray]:
    import os
    import requests

    mnist_path = "./mnist.npz"
    mnist_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"  # noqa
    if not os.path.exists(mnist_path):
        with open(mnist_path, "w+b") as f:
            f.write(requests.get(mnist_url, stream=True).content)
    res = np.load(mnist_path)
    return res["x_train"], res["y_train"], res["x_test"], res["y_test"]


def mnist_train(
        bottleneck: int,
        max_epoch: int,
        patience: int,
        ):
    x_train, _, x_test, _ = load_mnist()
    in_len = x_train[0].shape[0] * x_train[0].shape[0]
    x_train.resize(x_train.shape[0], in_len)
    x_test.resize(x_test.shape[0], in_len)
    x_train = x_train / 255
    x_test = x_test / 255
    autoencoder = Autoencoder(
        [in_len, bottleneck],
        [bottleneck, in_len],
        0.1,
        relu
    )
    autoencoder.train_dataset(
        x_train,
        max_epoch,
        patience,
        display_loss=True)
    autoencoder.save("autoencoder_mnist")


def mnist_test():
    x_train, _, x_test, _ = load_mnist()
    in_len = x_train[0].shape[0] * x_train[0].shape[0]
    img_shape = x_train[0].shape
    x_train.resize(x_train.shape[0], in_len)
    x_test.resize(x_test.shape[0], in_len)
    x_train = x_train / 255
    x_test = x_test / 255
    autoencoder = Autoencoder.load('autoencoder_mnist.npy')
    example: np.ndarray = x_test[np.random.randint(0, len(x_test))]
    output, _ = autoencoder.forward(example.flatten())
    plt.subplot(1, 2, 1)
    plt.matshow(example.reshape(img_shape), fignum=False)
    plt.subplot(1, 2, 2)
    plt.matshow(output.reshape(img_shape), fignum=False)
    plt.show()


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, nargs='?', default=50)
    parser.add_argument('-e', type=int, nargs='?', default=1000)
    parser.add_argument('-p', type=int, nargs='?', default=5)
    parser.add_argument('-r', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    if args.r:
        mnist_test()
    else:
        mnist_train(args.b, args.e, args.p)
        mnist_test()
