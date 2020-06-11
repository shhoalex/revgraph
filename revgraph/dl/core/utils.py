from .types import *


global_registry: Dict[str, Any] = {}


def get_graph(graph_instance: Metadata) -> rc.tensor:
    return graph_instance['graph']


def use_default(a: Any, b: Any) -> Any:
    return a if a is not None else b


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
