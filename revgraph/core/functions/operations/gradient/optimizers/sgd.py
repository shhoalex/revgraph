from revgraph.core.functions.operations.gradient.base.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, lr=0.001, decay=0.0):
        super().__init__()
        self.lr = lr
        self.decay = decay
        self.iteration = 0

    def update(self, param, memory, global_memory):
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iteration)) \
            if self.decay > 0 else self.lr
        param.data -= param.gradient * lr
        self.iteration += 1
        print(lr)
