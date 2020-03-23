from typing import Optional, Union

import numpy as np

from revgraph.core.computations.base.value import Value


class Variable(Value):
    def __init__(self,
                 data: Optional[Union[np.ndarray, list]]):
        super(Variable, self).__init__(data=data,
                                       requires_grad=True)
