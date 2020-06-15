from ..utils import *


def transform(f: TensorFunction) -> GraphBuilder:
    def graph_builder(prev_layer: Metadata) -> Metadata:
        metadata = {k: v for k, v in prev_layer.items() if k != 'graph'}
        metadata['graph'] = f(prev_layer['graph'])
        return metadata
    return graph_builder
