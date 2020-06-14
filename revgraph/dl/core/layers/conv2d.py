from ..utils import *


def conv2d(filters: int,
           kernel_size: Tuple[int, int],
           stride: Tuple[int, int] = (1, 1),
           padding: str = 'VALID',
           activation: ActivationFunction = 'linear',
           use_bias: bool = False,
           kernel_initializer: Initializer = 'glorot_normal',
           bias_initializer: Initializer = 'random_uniform',
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

    s0, s1 = (stride
              if isinstance(stride, tuple)
              else (stride, stride))
    k0, k1 = (kernel_size
              if isinstance(kernel_size, tuple)
              else (kernel_size, kernel_size))

    def graph_builder(prev_layer: Metadata) -> Metadata:
        metadata = {}
        graph = prev_layer['graph']
        h, w, c = prev_layer['units']

        # Initialize kernel as a tensor of shape (k0, k1, c, filters)
        v_kernel = rc.variable(kernel_initializer((k0, k1, c, filters)))
        metadata['v_kernel'] = v_kernel

        # Convolve the previous computational graph with the kernels
        graph = rc.conv2d(graph,
                          v_kernel,
                          padding=padding,
                          stride=stride)

        if use_bias:
            # Add bias units
            metadata['bias'] = v_bias = rc.variable(bias_initializer((1, 1, 1, filters,)))
            graph += v_bias

            if bias_regularizer is not None:
                metadata['regularized_bias'] = bias_regularizer(v_bias)

        metadata['graph'] = activation(graph)
        if kernel_regularizer is not None:
            metadata['regularized_kernel'] = kernel_regularizer(v_kernel)
        h_out = int(rc.conv2d.output_size(h, k0, padding, s0))
        w_out = int(rc.conv2d.output_size(w, k1, padding, s1))
        metadata['units'] = (filters, c, h_out, w_out)
        return metadata
    return graph_builder
