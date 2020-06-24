import dill as pkl

from .base_callback import BaseCallback, invoked_when


class Save(BaseCallback):
    def __init__(self,
                 path: str,
                 save_after_every: int = None,
                 save_after_all: bool = True):
        super().__init__()
        self.path = path
        self.save_after_every = save_after_every
        self.save_after_all = save_after_all

    def save_session(self):
        if not self.session.compiled:
            raise RuntimeError('Model is not compiled')
        with open(self.path, 'wb') as handler:
            pkl.dump({
                'prediction': self.session.prediction,
                'loss': self.session.loss,
                'optimization': self.session.optimization,
                'metrics': self.session.metrics
            }, handler)

    @invoked_when(lambda self: (
        (self.after_epoch and
         self.save_after_every is not None and
         (self.epoch+1) % self.save_after_every == 0)
    ))
    def save_method_during_training(self):
        self.output(f'    Saving session {self.session} to \'{self.path}\'')
        self.save_session()

    @invoked_when(lambda self: self.after_all and self.save_after_all)
    def save_method_after_training(self):
        self.output(f'Saving session {self.session} to \'{self.path}\'')
        self.save_session()
