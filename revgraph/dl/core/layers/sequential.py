from functools import reduce

from ..utils import *


def sequential(*layers: GraphBuilder) -> GraphBuilderNoParam:
    validate((len(layers) > 0, 'length of \'layers\' must be > 0'))

    def graph_builder() -> Metadata:
        nonlocal layers
        # Compose different layers into 1 big differentiable function
        layer, *layers = layers
        return reduce(lambda a, b: b(a), layers, layer())
    return graph_builder
