from typing import Optional, Union

import numpy as np

from revgraph.core.base.value import Value


class Variable(Value):
    """
    A variable represents a value with gradient (and therefore can be mutated/
    optimized).
    """
    def __init__(self,
                 data: Optional[Union[np.ndarray, list]]):
        super(Variable, self).__init__(data=data,
                                       requires_grad=True)
