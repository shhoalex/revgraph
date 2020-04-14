from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy

from revgraph.core.functions.operations.gradient.base.gradient import Gradient
from revgraph.core.values.variable import Variable


class Optimizer(Gradient):
    def __init__(self):
        super().__init__()
        self._objective = None
        self.params = None
        self.memory = None
        self.global_memory = None
        self.initialized = False

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, objective):
        self.args = self._objective = [objective]
        self.params = tuple(filter(lambda d: isinstance(d, Variable), objective.dependencies))
        self.global_memory = defaultdict(lambda: None)
        objective.register(self)
        self.dependencies = {self}.union(objective.dependencies)
        self.memory = defaultdict(lambda: defaultdict(lambda: None))
        self.initialized = False

    def minimize(self, objective):
        copied = deepcopy(self)
        copied.objective = objective
        return copied

    def backward(self, node):
        super().backward(node)

        if not self.initialized:
            self.init_global_memory(self.global_memory)
            for param in self.params:
                self.init_param_memory(param, self.memory[param], self.global_memory)
            self.initialized = True

        for param in self.params:
            self.update(param, self.memory[param], self.global_memory)

        self.clear_gradient()

    @abstractmethod
    def update(self, param, memory, global_memory):
        pass

    def init_global_memory(self, global_memory):
        pass

    def init_param_memory(self, param, memory, global_memory):
        pass
