from ..utils import *


def flatten() -> GraphBuilder:
    """
    dl.flatten layer builder
    """
    def graph_builder(prev_layer: Metadata) -> Metadata:
        metadata = {}
        init_regularized_nodes(metadata, prev_layer)
        graph = prev_layer['graph']
        metadata['units'] = (np.prod(prev_layer['units']),)
        metadata['graph'] = rc.flatten(graph)
        return metadata
    return graph_builder
