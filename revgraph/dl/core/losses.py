from .utils import *


@register
def categorical_cross_entropy(fold=rc.sum, eps=1e-20, axis=1):
    def function(y_true, y_pred):
        return fold(-rc.sum(y_true * rc.log(y_pred+eps) +
                    (1-y_true) * rc.log(1-y_pred+eps), axis=axis))
    return function


@register
def poisson(fold=rc.sum):
    def function(y_true, y_pred):
        return fold(y_pred - y_true * rc.log(y_pred))
    return function


@register
def kl_divergence(fold=rc.sum, eps=1e-20):
    def function(y_true, y_pred):
        return fold(y_true * rc.log(y_true / (y_pred + eps) + eps))
    return function


@register
def mean_squared_error(fold=rc.sum):
    def function(y_true, y_pred):
        return fold(rc.square(y_true - y_pred))
    return function


@register
def mean_absolute_error(fold=rc.sum):
    def function(y_true, y_pred):
        return fold(rc.abs(y_true - y_pred))
    return function


@register
def mean_absolute_percentage_error(fold=rc.sum):
    def function(y_true, y_pred):
        return fold(100 * rc.abs(y_true - y_pred) / y_true)
    return function


@register
def mean_squared_logarithmic_error(fold=rc.sum):
    def function(y_true, y_pred):
        return fold(rc.square(rc.log(y_true + 1.0) - rc.log(y_pred + 1.0)))
    return function


@register
def log_cosh(fold=rc.sum):
    def function(y_true, y_pred):
        x = y_true - y_pred
        return fold(rc.log(rc.cosh(x)))
    return function


@register
def hinge(fold=rc.sum):
    def function(y_true, y_pred):
        return fold(rc.max(1 - y_true * y_pred, 0))
    return function


@register
def squared_hinge(fold=rc.sum):
    def function(y_true, y_pred):
        return fold(rc.square(rc.max(1 - y_true * y_pred, 0)))
    return function
