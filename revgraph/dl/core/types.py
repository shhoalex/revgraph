from typing import *

import numpy as np

import revgraph.core as rc


TensorShape = Union[int, Iterable[Union[None, int]]]
Initializer = Union[str, Callable[[TensorShape], np.ndarray]]
TensorFunction = ActivationFunction = Regularizer = Union[str, Callable[[rc.tensor], rc.tensor]]
Metadata = Dict[str, Any]
Loss = Metric = Callable[[rc.tensor, rc.tensor], rc.tensor]
GraphBuilder = Callable[[Metadata], Metadata]
GraphBuilderNoParam = Callable[[], Metadata]
