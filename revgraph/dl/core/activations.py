from .utils import *


@register
def relu() -> ActivationFunction:
    def function(x: rc.tensor) -> rc.tensor:
        return (x>0) * x
    return function


@register
def tanh() -> ActivationFunction:
    return rc.tanh


@register
def sigmoid() -> ActivationFunction:
    def function(x: rc.tensor) -> rc.tensor:
        return 1 / (1 + rc.exp(-x))
    return function


@register
def softmax(axis=1, keepdims=True) -> ActivationFunction:
    def function(x: rc.tensor) -> rc.tensor:
        e_x = rc.exp(x)
        return e_x / e_x.sum(axis=axis, keepdims=keepdims)
    return function


@register
def softplus() -> ActivationFunction:
    def function(x: rc.tensor) -> rc.tensor:
        return rc.log(rc.exp(x) + 1.0)
    return function
