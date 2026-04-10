
import numpy as np


def softmax(v: np.ndarray) -> np.ndarray:
    v = v - np.max(v)
    exp_v = np.exp(v)
    return exp_v / np.sum(exp_v)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def regularize(v: np.ndarray) -> np.ndarray:
    v_min = v.min(axis=0)
    v_max = v.max(axis=0)
    if v_min - v_max == 0:
        return v
    return (v - v_min) / (v_max - v_min)


def interruptable(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            pass
    return inner
