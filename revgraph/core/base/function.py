from abc import ABC, abstractmethod
from typing import Union, Iterable, Dict, Any

import numpy as np

from .tensor import Tensor
from .value import Value


class Function(Tensor, ABC):
    """
    A Function represents a transformation from one tensor to another.
    """
    def __init__(self,
                 args: Iterable[Union[Tensor, list, int, float]] = None,
                 kwargs: Dict[str, Any] = None):
        """
        Note that gradient wouldn't be properly propagated when using kwargs
        (a bug). To fix it, simply implement a system for declaring
        parameters in a function (e.g. in f(a,b), the specification states that
        gradient only should be accumulated to b and a is just a normal
        parameter.
        """
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
        """
        Accumulate gradient FROM the reference to itself and then propagate
        the gradient to its child nodes.
        """
        super(Function, self).accumulate(reference, gradient)
        if self.context_completed():
            self.backward(*self.args, **self.kwargs)

    def new_context(self):
        """
        Initialize a new gradient accumulation context.
        """
        super(Function, self).new_context()
        for arg in self.args:
            arg.new_context()

    @staticmethod
    def match_shape(a, shape, axis):
        """
        Check whether the shape is "broadcasable".
        """
        if shape == () or a.shape == shape:
            return a
        axis = list(axis) if isinstance(axis, tuple) else axis
        new_shape = np.array(shape)
        new_shape[axis] = 1
        return np.broadcast_to(np.reshape(a, new_shape), shape)

    @abstractmethod
    def backward(self, *args, **kwargs) -> None:
        pass

    def clear_gradient(self):
        self.gradient = None
        for arg in self.args:
            arg.clear_gradient()
