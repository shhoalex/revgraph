import numpy as np

from revgraph.core.functions.operations.gradient.base.optimizer import Optimizer


class RMSProp(Optimizer):
    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-9, decay=0.0):
        super().__init__()
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay
        self.iteration = 0

    def init_param_memory(self, param, memory, global_memory):
        memory['accumulator'] = np.zeros_like(param.data)

    def before_update(self, global_memory):
        if self.decay > 0:
            global_memory['lr'] = self.lr * (1.0 / (1.0 + self.decay * self.iteration))
        else:
            global_memory['lr'] = self.lr
        self.iteration += 1

    def update(self, param, memory, global_memory):
        lr = global_memory['lr']
        memory['accumulator'] = self.rho * memory['accumulator'] + \
                                (1 - self.rho) * np.square(param.gradient)
        param.data = param.data - lr * param.gradient / \
                     np.sqrt(memory['accumulator'] + self.epsilon)
