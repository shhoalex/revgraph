from ..utils import *


def reshape(new_shape: TensorShape) -> GraphBuilder:
    def graph_builder(prev_layer: Metadata) -> Metadata:
        nonlocal new_shape
        graph = prev_layer['graph']
        old_shape = prev_layer['units']
        validate((np.prod(old_shape) == np.prod(new_shape),
                  f'{old_shape} cannot be reshaped to {new_shape}'))
        if isinstance(new_shape, int):
            new_shape = (new_shape,)
        return {
            'units': new_shape,
            'graph': rc.reshape(graph, (-1,) + new_shape)
        }
    return graph_builder
