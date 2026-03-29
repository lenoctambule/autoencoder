
import numpy as np
import matplotlib.pyplot as plt


def softmax(v: np.ndarray) -> np.ndarray:
    v = v - np.max(v)
    exp_v = np.exp(v)
    return exp_v / np.sum(exp_v)


def relu(x: np.ndarray, derivative=False) -> np.ndarray:
    if derivative:
        return x > 0
    return x * (x > 0)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def regularize(v: np.ndarray) -> np.ndarray:
    v_min = v.min(axis=0)
    v_max = v.max(axis=0)
    if v_min - v_max == 0:
        return v
    return (v - v_min) / (v_max - v_min)


def dynamic_loss_plot_init(losses: list):
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([0], losses, label="Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    return ax, line


def dynamic_loss_plot_update(ax, line, loss):
    line.set_xdata(range(len(loss)))
    line.set_ydata(loss)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.1)


def dynamic_loss_plot_finish(ax, line):
    plt.ioff()
    plt.show()
