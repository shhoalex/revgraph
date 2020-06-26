from ..utils import *


def inputs(shape: Union[Tuple[int, ...], int],
           input_placeholder_label: str = 'x') -> GraphBuilderNoParam:
    """
    dl.inputs layer builder

    Note that this must be the first layer of any architecture.
    """
    if isinstance(shape, int):
        shape = (shape,)
    validate((all(map(lambda n: n is not None, shape)),
              'shape must not consist non-int value'))

    def graph_builder() -> Metadata:
        return {
            'units': shape,
            'graph': rc.placeholder(shape=(None,) + shape,
                                    name=input_placeholder_label),
            'regularized_nodes': None
        }
    return graph_builder
