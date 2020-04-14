from typing import Dict, Union, Any, Iterable, Set, DefaultDict
from collections import defaultdict

import numpy as np

from revgraph.core.base.value import Value
from revgraph.core.base.function import Function
from revgraph.core.base.computation import Computation
from revgraph.core.values.placeholder import Placeholder


def get_placeholders(node: Computation) -> DefaultDict[Union[str, Computation], Computation]:
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


def run(node: Computation,
        feed_dict: Dict[Union[str, Placeholder], Union[Any]] = None) -> np.ndarray:
    feed_dict = feed_dict if feed_dict else {}
    dependencies = node.dependencies
    placeholders = get_placeholders(node)
    instantiate_placeholders(feed_dict, placeholders)
    new_backward_context(dependencies)

    node.forward()

    # call tuple() for strict evaluation
    result = node.data
    clear_placeholders(placeholders)
    return result


def new_backward_context(dependencies: Set[Computation]):
    for node in dependencies:
        node.ctx = defaultdict(lambda: 0)
        for p,n in node.references.items():
            if p in dependencies:
                node.ctx[p] += n
        node.ctx_counter = sum(node.ctx.values())


def instantiate_placeholders(feed_dict: Dict[Union[str, Placeholder], Union[Any]],
                             placeholders: DefaultDict[Union[str, Computation], Computation]):
    for c,v in feed_dict.items():
        node = c if isinstance(c, Placeholder) else placeholders[c]
        if node is None:
            continue
        elif isinstance(v, Value):
            node.feed(np.array(v.data))
        else:
            node.feed(np.array(v))


def clear_placeholders(placeholders: DefaultDict[Union[str, Computation], Computation]):
    for node in placeholders.values():
        if isinstance(node, Placeholder):
            node.clear_value()
