from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np

from .computation import Computation
from .value import Value


class Function(Computation, ABC):
    def __init__(self,
                 args: Union[List[Computation], list, int, float],
                 shape: Tuple[Optional[int], ...]):
        self.args = []
        requires_grad = False
        for arg in args:
            if not isinstance(arg, Computation):
                arg = Value(np.array(arg), requires_grad=False)
            if arg.requires_grad:
                arg.register(self)
                requires_grad = True
            self.args.append(arg)
        super(Function, self).__init__(shape, requires_grad)

    def accumulate(self, reference: Computation, gradient: np.ndarray):
        super(Function, self).accumulate(reference, gradient)
        if self.context_completed():
            self.backward()

    def new_context(self):
        super(Function, self).new_context()
        for arg in self.args:
            arg.new_context()

    @abstractmethod
    def backward(self) -> None:
        pass
