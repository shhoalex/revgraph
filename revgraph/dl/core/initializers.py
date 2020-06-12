from .utils import *


@register
def zeros() -> Initializer:
    def builder(shape: TensorShape) -> np.ndarray:
        return np.zeros_like(*shape)
    return builder


@register
def ones() -> Initializer:
    def builder(shape: TensorShape) -> np.ndarray:
        return np.ones_like(*shape)
    return builder


@register
def const(n: float = 1.0) -> Initializer:
    def builder(shape: TensorShape) -> np.ndarray:
        a = np.zeros_like(*shape)
        a.fill(n)
        return a
    return builder


@register
def random_normal(mean=0.0, stddev=0.05) -> Initializer:
    def builder(shape: TensorShape) -> np.ndarray:
        return stddev * np.random.randn(*shape) + mean
    return builder


@register
def random_uniform(minval=-0.05, maxval=0.05):
    def builder(shape: TensorShape) -> np.ndarray:
        return np.random.uniform(low=minval, high=maxval, size=shape)
    return builder


@register
def glorot_normal() -> Initializer:
    def builder(shape: TensorShape) -> np.ndarray:
        n_in, n_out = shape
        stddev = np.sqrt(2 / (n_in + n_out))
        return stddev * np.random.randn(*shape)
    return builder


@register
def glorot_uniform() -> Initializer:
    def builder(shape: TensorShape) -> np.ndarray:
        n_in, n_out = shape
        limit = np.sqrt(6 / (n_in + n_out))
        return np.random.uniform(low=-limit, high=limit, size=shape)
    return builder
