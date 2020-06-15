from .utils import *


@register
def l1(l1: float = 0.01) -> Regularizer:
    def function(x: rc.tensor) -> rc.tensor:
        return l1 * rc.sum(rc.abs(x))
    return function


@register
def l2(l2: float = 0.01) -> Regularizer:
    def function(x: rc.tensor) -> rc.tensor:
        return l2 * rc.sum(rc.square(x))
    return function


@register
def l1_l2(l1: float = 0.01,
          l2: float = 0.01) -> Regularizer:
    def function(x: rc.tensor) -> rc.tensor:
        return (l1 * rc.sum(rc.abs(x)) +
                l2 * rc.sum(rc.square(x)))
    return function
