from .types import *


global_registry: Dict[str, Any] = {}


def get_graph(graph_instance: Metadata) -> rc.tensor:
    return graph_instance['graph']


def append_regularized_nodes(metadata: Metadata, node: rc.tensor):
    metadata['regularized_nodes'] = (metadata['regularized_nodes'] + node
                                     if isinstance(metadata['regularized_nodes'], rc.tensor)
                                     else node)


def init_regularized_nodes(metadata: Metadata, prev_layer: Metadata):
    metadata['regularized_nodes'] = (prev_layer['regularized_nodes']
                                     if 'regularized_nodes' in prev_layer.keys()
                                     else None)


def use(key: str) -> Any:
    return global_registry[key]


def use_default(a: Any, b: Any) -> Any:
    return a if a is not None else b


def use_registry(key: Union[str, Any]) -> Any:
    if isinstance(key, str) and key in global_registry.keys():
        return global_registry[key]
    return key


def default_initializer(shape):
    return np.random.randn(*shape)


def use_default_initializer(initializer: Initializer) -> Initializer:
    return use_default(initializer, default_initializer)


def register(f: Callable) -> Callable:
    global_registry[f.__qualname__] = f()
    return f


def register_as(key: str) -> Callable:
    def decorator(f: Callable) -> Callable:
        global_registry[key] = f()
        return f
    return decorator


def validate(*predicates: Tuple[bool, str]) -> None:
    for valid, err_msg in predicates:
        if not valid:
            raise ValueError(err_msg)
