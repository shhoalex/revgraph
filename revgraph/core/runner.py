from typing import Dict, Union, Any, Iterable, List, Set, DefaultDict
from collections import defaultdict

import numpy as np

from revgraph.core.base.value import Value
from revgraph.core.base.function import Function
from revgraph.core.base.tensor import Tensor
from revgraph.core.values.placeholder import Placeholder


def get_placeholders(node: Tensor) -> DefaultDict[Union[str, Tensor], Tensor]:
    """
    Traverses the entire graph generates a dictionary that provides constant
    time access to the placeholder's reference.
    """
    stack = [node]
    namespace = defaultdict(lambda: None)
    while stack:
        head = stack.pop()
        if isinstance(head, Placeholder):
            namespace[head.name] = head
            namespace[head] = head
        elif isinstance(head, Function):
            stack.extend(head.args)
    return namespace


def run(node: Union[Tensor, Iterable[Tensor]],
        feed_dict: Dict[Union[str, Placeholder], Union[Any]] = None) -> Union[np.ndarray, List[np.array]]:
    """
    Execute the node with the placeholder being substitute by the values in
    feed_dict.
    """
    if not isinstance(node, Tensor):
        return run_many(node, feed_dict)
    feed_dict = feed_dict if feed_dict else {}
    dependencies = node.dependencies
    placeholders = get_placeholders(node)
    instantiate_placeholders(feed_dict, placeholders)
    new_backward_context(dependencies)

    node.forward()

    # reference note.data before cleanup
    result = node.data
    clear_placeholders(placeholders)
    return result


def run_many(nodes: Iterable[Tensor],
             feed_dict: Dict[Union[str, Placeholder], Union[Any]] = None) -> List[np.ndarray]:
    """
    Shorthand for running multiple nodes.
    """
    return [run(node, feed_dict) for node in nodes]


def new_backward_context(dependencies: Set[Tensor]):
    """
    Generate a dictionary for creating a "gradient context".
    """
    for node in dependencies:
        node.ctx = defaultdict(lambda: 0)
        for p,n in node.references.items():
            if p in dependencies:
                node.ctx[p] += n
        node.ctx_counter = sum(node.ctx.values())


def instantiate_placeholders(feed_dict: Dict[Union[str, Placeholder], Union[Any]],
                             placeholders: DefaultDict[Union[str, Tensor], Tensor]):
    """
    Helper function for substituting placeholders in a graph with the values in
    feed_dict.
    """
    for c,v in feed_dict.items():
        node = c if isinstance(c, Placeholder) else placeholders[c]
        if node is None:
            continue
        elif isinstance(v, Value):
            node.feed(np.array(v.data))
        else:
            node.feed(np.array(v))


def clear_placeholders(placeholders: DefaultDict[Union[str, Tensor], Tensor]):
    for node in placeholders.values():
        if isinstance(node, Placeholder):
            node.clear_value()
