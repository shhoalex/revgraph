from typing import Dict, Union, Any

import numpy as np

from revgraph.core.base.value import Value
from revgraph.core.base.function import Function
from revgraph.core.base.computation import Computation
from revgraph.core.values.placeholder import Placeholder


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
