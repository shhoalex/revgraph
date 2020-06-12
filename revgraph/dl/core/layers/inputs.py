from ..utils import *


def inputs(shape: Tuple[int],
           input_placeholder_label: str = 'x') -> GraphBuilder:
    if isinstance(shape, int):
        shape = (shape,)
    validate((all(map(lambda n: n is not None, shape)),
              'shape must not consist non-int value'))

    def graph_builder() -> Metadata:
        return {
            'units': np.prod(shape),
            'graph': rc.placeholder(shape=(None,) + shape,
                                    name=input_placeholder_label)
        }
    return graph_builder
