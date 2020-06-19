from .base_callback import BaseCallback, invoked_when


class Progress(BaseCallback):
    def __init__(self):
        super().__init__()

    @invoked_when(lambda self: (
        self.batch == 0 and
        self.before_execution
    ))
    def output_epoch(self):
        self.output(f'Epoch {self.epoch}')
