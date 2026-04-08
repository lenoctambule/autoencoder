import matplotlib.pyplot as plt
import numpy as np
import os
import signal
from easyvae.autoencoder import ( # noqa
        VariationalAutoencoder,
        ClassicalAutoencoder,
        AAutoencoder
    )
from easyvae.activations import LeakyReLU


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
        cls: type[AAutoencoder],) -> AAutoencoder:
    x_train, _, x_test, _ = load_mnist()
    in_len = x_train[0].shape[0] * x_train[0].shape[0]
    x_train.resize(x_train.shape[0], in_len)
    x_test.resize(x_test.shape[0], in_len)
    x_train = x_train / 255
    if os.path.exists(filename):
        autoencoder = cls.load(filename)
    else:
        autoencoder = cls(
            [in_len, 256, 2],
            [2, 256, in_len],
            0.001,
            LeakyReLU()
        )

    def handler(signum, frame):
        print(f"Saving {filename} before exit ...")
        autoencoder.save(filename)
        plt.close()
        plt.ioff()
        mnist_test(autoencoder)
        exit()

    signal.signal(signal.SIGINT, handler)
    print("CTRL+C to exit and save model.")
    autoencoder.train_dataset(
        x_train,
        max_epoch,
        patience,
        display_loss=True)
    autoencoder.save(filename)
    print("Training complete !")
    return autoencoder


def mnist_test(model: str | AAutoencoder):
    x_train, _, x_test, y_test = load_mnist()
    in_len = x_train[0].shape[0] * x_train[0].shape[0]
    img_shape = x_train[0].shape
    x_train.resize(x_train.shape[0], in_len)
    x_test.resize(x_test.shape[0], in_len)
    x_train = x_train / 255
    x_test = x_test / 255
    if isinstance(model, str):
        autoencoder: AAutoencoder = AAutoencoder.load(model)
    else:
        autoencoder = model
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
    code = np.reshape(code, (code.shape[0], 1))
    plt.matshow(code, fignum=False)
    plt.title(f"Code ({y_test[idx]})")
    plt.show()
    if code.shape[0] == 2:
        codes = []
        for x in x_test:
            _, c = autoencoder.forward(x.flatten())
            codes.append(c)
        codes = np.array(codes)
        if codes.shape[1] == 2:
            plt.figure(figsize=(6, 6))
            scatter = plt.scatter(
                codes[:, 0],
                codes[:, 1],
                c=y_test,
                cmap='tab10',
                s=5,
                alpha=0.7
            )
            plt.colorbar(scatter)
            plt.grid(True)
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
        autoencoder = mnist_train(
            args.m,
            args.e,
            args.p,
            VariationalAutoencoder
        )
        mnist_test(autoencoder)
