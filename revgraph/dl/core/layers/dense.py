from ..utils import *


def dense(units: int,
          use_bias: bool = True,
          activation: ActivationFunction = 'linear',
          kernel_initializer: Initializer = 'glorot_normal',
          bias_initializer: Initializer = 'glorot_normal',
          kernel_regularizer: Regularizer = None,
          bias_regularizer: Regularizer = None,
          activity_regularizer: Regularizer = None,
          kernel_constraint: Constraint = None,
          bias_constraint: Constraint = None
          ) -> GraphBuilder:
    validate((units > 0,
              f'\'units\' must be a positive integer, instead of {units}'),
             (callable(activation) or activation is None or isinstance(activation, str),
              f'\'activation\' must be callable, instead of type {type(activation)}'))

    activation = use_default(use_registry(activation), lambda x: x)
    kernel_initializer = use_default_initializer(use_registry(kernel_initializer))
    bias_initializer = use_default_initializer(use_registry(bias_initializer))

    def graph_builder(prev_layer: Metadata) -> Metadata:
        metadata = {
            'units': units,
            'use_bias': use_bias
        }
        init_regularized_nodes(metadata, prev_layer)

        # Create kernel
        init_weights = kernel_initializer((np.prod(prev_layer['units']), units))
        metadata['kernel'] = v_kernel = rc.variable(init_weights)

        graph = prev_layer['graph'].dot(v_kernel)
        if use_bias:
            # Create bias
            init_bias = bias_initializer((1, units))
            metadata['bias'] = v_bias = rc.variable(init_bias)
            graph += v_bias
            if bias_regularizer is not None:
                r_bias = bias_regularizer(v_bias)
                append_regularized_nodes(metadata, r_bias)

        # Apply activation function
        graph = activation(graph)

        if kernel_regularizer is not None:
            # Apply kernel regularization
            r_kernel = kernel_regularizer(v_kernel)
            append_regularized_nodes(metadata, r_kernel)

        if activity_regularizer is not None:
            # Apply regularization to graph
            r_output = activity_regularizer(graph)
            append_regularized_nodes(metadata, r_output)
        metadata['graph'] = graph
        return metadata
    return graph_builder
