from typing import *

import numpy as np

import revgraph.core as rc


TensorShape = Union[int, Iterable[Union[None, int]]]
Initializer = Union[str, Callable[[TensorShape], np.ndarray]]
TensorFunction = Union[str, Callable[[rc.tensor], rc.tensor]]
Loss = Metric = ActivationFunction = Regularizer = TensorFunction
Metadata = Dict[str, Any]
GraphBuilder = Callable[[Metadata], Metadata]
GraphBuilderNoParam = Callable[[], Metadata]
