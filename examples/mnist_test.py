import matplotlib.pyplot as plt
import numpy as np
import os
from easyvae.autoencoder import ( # noqa
        VariationalAutoencoder,
        ClassicalAutoencoder,
        LabelingVAE,
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
    x_train, _, _, _ = load_mnist()
    in_len = x_train[0].shape[0] * x_train[0].shape[0]
    x_train.resize(x_train.shape[0], in_len)
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
    print("CTRL+C to interrupt training.")
    autoencoder.train_dataset(
        x_train,
        max_epoch,
        patience,
        display_loss=True)
    autoencoder.save(filename)
    print("Training complete !")
    return autoencoder


def plot_mnist_latent_space(autoencoder: AAutoencoder, x: np.ndarray, y,):
    codes = []
    for x in x:
        _, c = autoencoder.forward(x.flatten())
        codes.append(c)
    codes = np.array(codes)
    if codes.shape[1] == 2:
        plt.figure(figsize=(6, 6))
        scatter = plt.scatter(
            codes[:, 0],
            codes[:, 1],
            c=y,
            cmap='tab10',
            s=5,
            alpha=0.7
        )
        plt.colorbar(scatter)
        plt.grid(True)
        plt.show()


def plot_random_reconstruction(
        autoencoder: AAutoencoder,
        example: np.ndarray,
        img_shape,
        y):
    output, code = autoencoder.forward(example.flatten())
    plt.subplot(1, 2, 1)
    plt.matshow(
        example.reshape(img_shape),
        fignum=False)
    plt.title(f"Input ({y})")
    plt.subplot(1, 2, 2)
    plt.matshow(
        output.reshape(img_shape),
        fignum=False)
    plt.title(f"Output ({y})")
    print(f'{code.tolist()}')


def mnist_test(model: str | AAutoencoder | LabelingVAE):
    x_train, y_train, x_test, y_test = load_mnist()
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
    print("Testing model ...\n")
    print(autoencoder)
    idx = np.random.randint(0, len(x_test))
    example: np.ndarray = x_test[idx]
    y_train = [str(int(i)) for i in y_train]
    autoencoder.learn_labels(x_train, y_train, 5)
    res = autoencoder.label(x_train[idx])
    for k, v in res.items():
        print(f"{k} => {v}")
    plot_random_reconstruction(autoencoder, example, img_shape, y_test[idx])
    if autoencoder.space_dim == 2:
        plot_mnist_latent_space(autoencoder, x_test, y_test)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e',
        type=int,
        nargs='?',
        default=30,
        help='Max epochs'
    )
    parser.add_argument(
        '-p',
        type=int,
        nargs='?',
        default=30,
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
        help='Run the model'
    )
    args = parser.parse_args(sys.argv[1:])
    if args.r:
        mnist_test(args.m)
    else:
        autoencoder = mnist_train(
            args.m,
            args.e,
            args.p,
            LabelingVAE
        )
        mnist_test(autoencoder)
