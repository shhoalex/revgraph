from typing import *

import numpy as np

import revgraph.core as rc


TensorShape = Union[int, Iterable[Union[None, int]]]
Initializer = Union[str, Callable[[TensorShape], np.ndarray]]
TensorFunction = ActivationFunction = Regularizer = Union[str, Callable[[rc.tensor], rc.tensor]]
Constraint = Union[str, Callable[[np.ndarray], np.ndarray]]
Metadata = Dict[str, Any]
GraphBuilder = Callable[[Metadata], Metadata]
GraphBuilderNoParam = Callable[[], Metadata]
