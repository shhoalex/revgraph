from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from .computation import Computation


class Function(Computation, ABC):
    def __init__(self,
                 args: List[Computation],
                 shape: Tuple[Optional[int], ...],
                 requires_grad: bool = False):
        super(Function, self).__init__(shape, requires_grad)
        self.args = args

        for arg in self.args:
            if arg.requires_grad:
                arg.register(self)

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
