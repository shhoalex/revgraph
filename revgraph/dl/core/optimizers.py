from .utils import *


@register
def sgd(lr: float = 0.01,
        momentum: float = 0.0,
        decay: float = 0.0,
        nesterov: bool = False):
    return rc.sgd(lr, momentum, decay, nesterov).minimize


@register
def adagrad(lr: float = 0.001,
            epsilon: float = 1e-9,
            decay: float = 0.0):
    return rc.adagrad(lr, epsilon, decay).minimize


@register
def adadelta(lr: float = 1.0,
             rho: float = 0.95,
             epsilon: float = 1e-6,
             decay: float = 0.0):
    return rc.adadelta(lr, rho, epsilon, decay).minimize


@register
def rmsprop(lr: float = 0.001,
            rho: float = 0.9,
            epsilon: float = 1e-9,
            decay: float = 0.0):
    return rc.rmsprop(lr, rho, epsilon, decay).minimize


@register
def adam(lr: float = 0.001,
         beta1: float = 0.9,
         beta2: float = 0.999,
         amsgrad: float = False,
         epsilon: float = 1e-6,
         decay: float = 0.0):
    return rc.adam(lr, beta1, beta2, amsgrad, epsilon, decay).minimize
