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
        feed_dict: Dict[Union[str, Placeholder], Union[Any]] = None,
        results: Iterable[Computation] = None) -> Iterable[np.ndarray]:
    feed_dict = feed_dict if feed_dict else {}
    results = results if results else [node]
    dependencies = node.dependencies
    placeholders = get_placeholders(node)
    instantiate_placeholders(feed_dict, placeholders)
    new_backward_context(dependencies)

    node.forward()

    # call tuple() for strict evaluation
    result = tuple(map(lambda n: n.data, results))
    clear_placeholders(placeholders)
    return result


def new_backward_context(dependencies: Set[Computation]):
    for node in dependencies:
        node.ctx = defaultdict(lambda: 0)
        for p,n in node.references.items():
            if p in dependencies:
                node.ctx[p] += 1
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

"""
class Runner:
    def __init__(self, node: Computation):
        self.node = node

    def __repr__(self):
        return f'Runner(node={self.node})'

    @property
    def node(self):
        return self._node

    @node.setter
    def node(self, node):
        self._node = node
        stack = [self._node]
        self.placeholders = {}
        while stack:
            head = stack.pop()
            if isinstance(head, Placeholder):
                self.placeholders[head] = head
                self.placeholders[head.name] = head
            elif isinstance(head, Function):
                stack.extend(head.args)

    @node.deleter
    def node(self):
        self.placeholders = {}
        del self._node

    def run(self, feed_dict: Dict[Union[str, Placeholder], Union[Any]] = None):
        self.instantiate_placeholder(feed_dict)
        self.node.new_context()
        result = self.node.forward()
        self.clear_placeholder(feed_dict)
        return result

    def instantiate_placeholder(self, feed_dict: Dict[Union[str, Placeholder], Union[Any]] = None):
        if feed_dict is not None:
            for c,v in feed_dict.items():
                node = c if isinstance(c, Placeholder) else self.placeholders[c]
                if isinstance(v, Value):
                    node.feed(np.array(v.data))
                else:
                    node.feed(np.array(v))

    def clear_placeholder(self, feed_dict: Dict[Union[str, Placeholder], Union[Any]] = None):
        if feed_dict is not None:
            for c in feed_dict.keys():
                node = c if isinstance(c, Placeholder) else self.placeholders[c]
                node.clear_value()
"""
