from typing import *

import numpy as np

import revgraph.core as rc


TensorShape = Union[int, Iterable[Union[None, int]]]
ActivationFunction = Callable[[rc.tensor], rc.tensor]
Initializer = Callable[[TensorShape], np.ndarray]
Regularizer = Callable[[rc.tensor], rc.tensor]
Constraint = Callable[[np.ndarray], np.ndarray]
Metadata = Dict[str, Any]
GraphBuilder = Callable[[Metadata], Metadata]
