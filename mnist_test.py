import matplotlib.pyplot as plt
import numpy as np
from autoencoder import ClassicalAutoencoder
from activations import LeakyReLU
import os


def load_mnist() -> list[np.ndarray]:
    import requests

    mnist_path = "./mnist.npz"
    mnist_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"  # noqa
    if not os.path.exists(mnist_path):
        with open(mnist_path, "w+b") as f:
            f.write(requests.get(mnist_url, stream=True).content)
    res = np.load(mnist_path)
    return res["x_train"], res["y_train"], res["x_test"], res["y_test"]


def mnist_train(
        filename: str,
        max_epoch: int,
        patience: int,
        ) -> ClassicalAutoencoder:
    x_train, _, x_test, _ = load_mnist()
    in_len = x_train[0].shape[0] * x_train[0].shape[0]
    x_train.resize(x_train.shape[0], in_len)
    x_test.resize(x_test.shape[0], in_len)
    x_train = x_train / 255
    x_test = x_test / 255
    if os.path.exists(filename):
        autoencoder = ClassicalAutoencoder.load(filename)
    else:
        autoencoder = ClassicalAutoencoder(
            [in_len, 64, 16],
            [16, 64, in_len],
            0.01,
            LeakyReLU()
        )
    autoencoder.train_dataset(
        x_train,
        max_epoch,
        patience,
        display_loss=True)
    autoencoder.save(filename)
    return autoencoder


def mnist_test(model: str | ClassicalAutoencoder):
    x_train, _, x_test, y_test = load_mnist()
    in_len = x_train[0].shape[0] * x_train[0].shape[0]
    img_shape = x_train[0].shape
    x_train.resize(x_train.shape[0], in_len)
    x_test.resize(x_test.shape[0], in_len)
    x_train = x_train / 255
    x_test = x_test / 255
    if isinstance(model, str):
        autoencoder: ClassicalAutoencoder = ClassicalAutoencoder.load(model)
    else:
        autoencoder = model
    print(autoencoder)
    idx = np.random.randint(0, len(x_test))
    example: np.ndarray = x_test[idx]
    output, code = autoencoder.forward(example.flatten())
    plt.subplot(1, 3, 1)
    plt.matshow(
        example.reshape(img_shape),
        fignum=False)
    plt.title(f"Input ({y_test[idx]})")
    plt.subplot(1, 3, 2)
    plt.matshow(
        output.reshape(img_shape),
        fignum=False)
    plt.title(f"Output ({y_test[idx]})")
    plt.subplot(1, 3, 3)
    s = int(np.ceil(np.sqrt(code.shape[0])))
    code.resize((s, s), refcheck=False)
    plt.matshow(code, fignum=False)
    plt.title(f"Code ({y_test[idx]})")
    plt.show()


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e',
        type=int,
        nargs='?',
        default=1000,
        help='Max epochs'
    )
    parser.add_argument(
        '-p',
        type=int,
        nargs='?',
        default=5,
        help='Patience'
    )
    parser.add_argument(
        '-m',
        type=str, nargs='?',
        default='autoencoder_mnist.npy',
        help='Model filename to save in run mode or load in training mode'
    )
    parser.add_argument(
        '-r',
        action='store_true',
        help='Run mode'
    )
    args = parser.parse_args(sys.argv[1:])
    if args.r:
        mnist_test(args.m)
    else:
        autoencoder = mnist_train(args.m, args.e, args.p)
        mnist_test(autoencoder)
