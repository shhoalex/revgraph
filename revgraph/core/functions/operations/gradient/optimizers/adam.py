import numpy as np

from revgraph.core.functions.operations.gradient.base.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 amsgrad=False,
                 epsilon=1e-6,
                 decay=0.0):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
        self.amsgrad = amsgrad
        self.iteration = 0

    def init_param_memory(self, param, memory, global_memory):
        memory['m'] = np.zeros_like(param.data)
        memory['v'] = np.zeros_like(param.data)

        if self.amsgrad:
            memory['vhat'] = np.zeros_like(param.data)
        else:
            memory['vhat'] = np.zeros(1)

    def before_update(self, global_memory):
        self.iteration += 1
        if self.decay > 0:
            lr = global_memory['lr'] = self.lr * (1.0 / (1.0 + self.decay *
                                                         self.iteration))
        else:
            lr = global_memory['lr'] = self.lr

        t = float(self.iteration) + 1
        global_memory['lr_t'] = lr * (np.sqrt(1 - np.power(self.beta2, t)) /
                                      (1 - np.power(self.beta1, t)))

    def update(self, param, memory, global_memory):
        lr, lr_t = map(global_memory.__getitem__, ['lr', 'lr_t'])
        m, v, vhat = map(memory.__getitem__, ['m', 'v', 'vhat'])
        b1, b2 = self.beta1, self.beta2
        g = param.gradient
        m_t = memory['m'] = (b1 * m) + (1 - b1) * g
        v_t = memory['v'] = (b2 * v) + (1 - b2) * np.square(g)

        if self.amsgrad:
            vhat_t = memory['vhat'] = np.maximum(vhat, v_t)
            param.data -= lr_t * m_t / (np.sqrt(vhat_t) + self.epsilon)
        else:
            param.data -= lr_t * m_t / (np.sqrt(v_t) + self.epsilon)
