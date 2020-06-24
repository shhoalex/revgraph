from inspect import getmembers
from typing import Any, Callable, IO

import numpy as np


class BaseCallback(object):
    def __init__(self):
        self.callbacks = []

        # Hacky way to bind callback condition to object
        # Similar to how generic_function works
        for _, m in getmembers(self):
            if hasattr(m, 'callback_cond'):
                self.callbacks.append([m.callback_cond, m])
        self.epoch = None
        self.n_epochs = None
        self.batch = None
        self.n_batches = None
        self.before_batch = None
        self.after_batch = None
        self.session = None
        self.output = None
        self.before_all = None
        self.after_all = None
        self.before_epoch = None
        self.after_epoch = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_validation = None
        self.y_validation = None
        self.x_batch = None
        self.y_batch = None

    def send_metadata(self,
                      epoch: int,
                      n_epochs: int,
                      batch: int,
                      n_batches: int,
                      before_batch: bool,
                      after_batch: bool,
                      before_all: bool,
                      after_all: bool,
                      before_epoch: bool,
                      after_epoch: bool,
                      session: 'Session',
                      output: Callable[[Any], IO[None]],
                      x_train: np.array = None,
                      y_train: np.array = None,
                      x_test: np.array = None,
                      y_test: np.array = None,
                      x_validation: np.array = None,
                      y_validation: np.array = None,
                      x_batch: np.array = None,
                      y_batch: np.array = None) -> None:
        self.epoch = epoch
        self.n_epochs = n_epochs
        self.batch = batch
        self.n_batches = n_batches
        self.before_batch = before_batch
        self.after_batch = after_batch
        self.before_all = before_all
        self.after_all = after_all
        self.before_epoch = before_epoch
        self.after_epoch = after_epoch
        self.session = session
        self.output = output
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.x_batch = x_batch
        self.y_batch = y_batch

    def invoke(self):
        for should_invoke, invokable in self.callbacks:
            if should_invoke(self):
                invokable()


def invoked_when(cond: Callable[[BaseCallback], Any]):
    def decorator(function: Callable[[BaseCallback], IO[None]]):
        function.callback_cond = cond
        return function
    return decorator
