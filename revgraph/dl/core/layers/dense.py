from ..utils import *


def dense(units: int,
          use_bias: bool = True,
          activation: ActivationFunction = None,
          kernel_initializer: Initializer = None,
          bias_initializer: Initializer = None,
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
        # Create kernel
        init_weights = kernel_initializer((prev_layer['units'], units))
        metadata['kernel'] = v_kernel = rc.variable(init_weights)

        graph = prev_layer['graph'].dot(v_kernel)
        if use_bias:
            # Create bias
            init_bias = bias_initializer((1, units))
            metadata['bias'] = v_bias = rc.variable(init_bias)
            graph += v_bias
            if bias_regularizer is not None:
                metadata['regularized_bias'] = bias_regularizer(v_bias)

        # Apply activation function
        graph = activation(graph)

        if kernel_regularizer is not None:
            # Apply kernel regularization
            metadata['regularized_kernel'] = kernel_regularizer(v_kernel)

        if activity_regularizer is not None:
            # Apply regularization to graph
            metadata['regularized_output'] = activity_regularizer(graph)
        metadata['graph'] = graph
        return metadata
    return graph_builder
