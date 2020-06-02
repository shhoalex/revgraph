import numpy as np

from revgraph.core.functions.operations.gradient.base.optimizer import Optimizer


class AdaDelta(Optimizer):
    def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6, decay=0.0):
        super().__init__()
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay
        self.iteration = 0

    def init_param_memory(self, param, memory, global_memory):
        memory['accumulator'] = np.zeros_like(param.data)
        memory['delta_accumulator'] = np.zeros_like(param.data)

    def before_update(self, global_memory):
        if self.decay > 0:
            global_memory['lr'] = self.lr * (1.0 / (1.0 + self.decay * self.iteration))
        else:
            global_memory['lr'] = self.lr
        self.iteration += 1

    def update(self, param, memory, global_memory):
        lr = global_memory['lr']
        g = param.gradient
        delta_accumulator = memory['delta_accumulator']
        memory['accumulator'] = self.rho * memory['accumulator'] + \
                                (1 - self.rho) * np.square(g)
        d = g * np.sqrt(delta_accumulator + self.epsilon) / \
            np.sqrt(memory['accumulator'] + self.epsilon)
        param.data -= lr * d
        memory['delta_accumulator'] = self.rho * delta_accumulator + \
                                      (1 - self.rho) * np.square(d)
