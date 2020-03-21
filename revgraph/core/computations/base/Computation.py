from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Tuple, Optional

import numpy as np


class Computation(ABC):
    def __init__(self,
                 shape: Optional[Tuple[int, ...]] = None,
                 requires_grad: bool = False):
        self.gradient = None
        self.shape = shape
        self.requires_grad = requires_grad
        self.references = defaultdict(lambda: 0)
        self.current_count = 0
        self.ctx = None

    def new_session(self):
        self.ctx = self.references.copy()

    def accumulate(self, reference: 'Computation', gradient: np.array):
        if self.ctx[reference] <= 0:
            raise ValueError('Invalid node for gradient propagation')
        else:
            self.ctx[reference] -= 1
        if self.shape is None:
            self.shape = gradient.shape
            self.gradient = gradient
        else:
            assert gradient.shape == self.gradient.shape
            self.gradient += gradient

    def register(self, reference: 'Computation'):
        # Add a permanent computational node (independent of context)
        self.references[reference] += 1

    @abstractmethod
    def forward(self):
        pass
