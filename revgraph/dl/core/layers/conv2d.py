from ..utils import *


def conv2d(filters: int,
           kernel_size: Tuple[int, int],
           stride: Tuple[int, int] = (1, 1),
           padding: str = 'VALID',
           activation: ActivationFunction = None,
           use_bias: bool = False,
           kernel_initializer: Initializer = None,
           bias_initializer: Initializer = None,
           kernel_regularizer: Regularizer = None,
           bias_regularizer: Regularizer = None,
           activity_regularizer: Regularizer = None,
           kernel_constraint: Constraint = None,
           bias_constraint: Constraint = None) -> GraphBuilder:
    validate((filters > 0,
              f'\'filters\' must be a positive integer, instead of {filters}'),
             (callable(activation) or activation is None or isinstance(activation, str),
              f'\'activation\' must be callable, instead of type {type(activation)}'))

    activation = use_default(use_registry(activation), lambda x: x)
    kernel_initializer = use_default_initializer(use_registry(kernel_initializer))
    bias_initializer = use_default_initializer(use_registry(bias_initializer))

    def graph_builder(prev_layer: Metadata) -> Metadata:
        metadata = {}
        graph = prev_layer['graph']
        h, w, c = prev_layer['units']
        v_kernel = rc.variable(kernel_initializer((*kernel_size, c, filters)))
        graph = rc.conv2d(graph,
                          v_kernel,
                          padding=padding,
                          stride=stride)
        metadata['graph'] = activation(graph)
        if kernel_regularizer is not None:
            metadata['regularized_kernel'] = kernel_regularizer(v_kernel)
        metadata['units'] = (filters, c, *kernel_size)
        return metadata
    return graph_builder
