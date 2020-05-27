from abc import ABC, abstractmethod
from typing import Union, Iterable, Dict, Any

import numpy as np

from .tensor import Tensor
from .value import Value


class Function(Tensor, ABC):
    def __init__(self,
                 args: Iterable[Union[Tensor, list, int, float]] = None,
                 kwargs: Dict[str, Any] = None):
        args = [] if args is None else args
        kwargs = {} if kwargs is None else kwargs
        self.args = []
        self.data = None
        requires_grad = False
        for arg in args:
            if not isinstance(arg, Tensor):
                arg = Value(np.array(arg), requires_grad=False)
            if arg.requires_grad:
                arg.register(self)
                requires_grad = True
            self.args.append(arg)
        self.kwargs = kwargs
        self.dependencies = {self}.union(*map(lambda n: n.dependencies, self.args))
        super(Function, self).__init__(requires_grad=requires_grad)

    def accumulate(self, reference: Tensor, gradient: np.ndarray):
        super(Function, self).accumulate(reference, gradient)
        if self.context_completed():
            self.backward(*self.args, **self.kwargs)

    def new_context(self):
        super(Function, self).new_context()
        for arg in self.args:
            arg.new_context()

    @abstractmethod
    def backward(self, *args, **kwargs) -> None:
        pass

    def clear_gradient(self):
        self.gradient = None
        for arg in self.args:
            arg.clear_gradient()
