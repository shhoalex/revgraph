from .types import *


def get_graph(graph_instance: Metadata) -> rc.tensor:
    return graph_instance['graph']


def use_default(a: Any, b: Any) -> Any:
    return a if a is not None else b


def default_initializer(shape):
    return np.random.randn(*shape)


def use_default_initializer(initializer: Initializer) -> Initializer:
    return use_default(initializer, default_initializer)


def validate(*predicates: Tuple[bool, str]) -> None:
    for valid, err_msg in predicates:
        if not valid:
            raise ValueError(err_msg)
