from time import time

from .base_callback import BaseCallback, invoked_when


class Progress(BaseCallback):
    def __init__(self):
        super().__init__()

    @invoked_when(lambda self: self.before_all)
    def before_all(self):
        self.output(f'Training Model {self.session.prediction} with {len(self.x_train)} records')
        self.start_time = time()

    @invoked_when(lambda self: self.before_epoch)
    def before_epoch(self):
        self.output(f'  + Epoch {self.epoch+1}/{self.n_epochs}')
        self.epoch_start_time = time()

    @invoked_when(lambda self: self.before_batch)
    def before_batch(self):
        percentage = (100 * (self.batch+1)) / self.n_batches
        n = min(int(percentage // 10) + 1, 10)
        self.output(f'    [{"#" * (n-1)}>{" " * (10-n)}] ({percentage:.1f}%)', end='\r')

    @invoked_when(lambda self: self.after_epoch)
    def after_epoch(self):
        percentage = (100 * (self.batch + 1)) / self.n_batches
        n = min(int(percentage // 10) + 1, 10)
        self.output(f'    [{"#" * n}{" " * (10-n)}] ({percentage:.1f}%)')
        self.epoch_end_time = time()
        self.output(f'    Time Elapsed: {self.epoch_end_time - self.epoch_start_time:.3f}s')

    @invoked_when(lambda self: self.after_all)
    def after_all(self):
        self.end_time = time()
        self.output(f'Training Completed.\nTotal Time Elapsed: {self.end_time - self.start_time:.3f}s')
