import numpy as np

from revgraph.core.functions.operations.gradient.base.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, lr=0.001, momentum=0.0, decay=0.0, nesterov=False):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.iteration = 0
        self.nesterov = nesterov

    def init_param_memory(self, param, memory, global_memory):
        memory['velocity'] = np.zeros_like(param.data)

    def before_update(self, global_memory):
        if self.decay > 0:
            global_memory['lr'] = self.lr * (1.0 / (1.0 + self.decay * self.iteration))
        else:
            global_memory['lr'] = self.lr
        self.iteration += 1

    def update(self, param, memory, global_memory):
        lr = global_memory['lr']
        v = memory['velocity'] = self.momentum * memory['velocity'] - lr * param.gradient
        param.data += self.momentum * v - lr * param.gradient if self.nesterov else v
