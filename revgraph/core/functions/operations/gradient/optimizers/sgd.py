from revgraph.core.functions.operations.gradient.base.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, lr=0.001):
        super().__init__()
        self.lr = lr

    def update(self, param, memory, global_memory):
        param.data -= param.gradient * self.lr
