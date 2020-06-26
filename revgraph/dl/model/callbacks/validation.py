from typing import Any, Iterable

from .base_callback import BaseCallback, invoked_when


class Validation(BaseCallback):
    """
    Callback for evaluating the validation set after every "after_every"
    epochs.
    """
    def __init__(self,
                 after_every: int = 1,
                 x_validation: Iterable[Any] = None,
                 y_validation: Iterable[Any] = None):
        super().__init__()
        self.period = after_every
        self.warned = False
        self.preset_x_validation = x_validation
        self.preset_y_validation = y_validation

    @invoked_when(lambda self: self.after_epoch and (self.epoch+1) % self.period == 0)
    def evaluate_validation_set(self):
        if self.preset_x_validation is not None and self.preset_y_validation is not None:
            self.x_validation = self.preset_x_validation
            self.y_validation = self.preset_y_validation
        if not self.warned and len(self.x_validation) == 0:
            self.output('WARNING: length of validation set is 0.')
            self.warned = True
        metrics = self.session.evaluate_metrics(self.x_validation, self.y_validation)
        self.output('\n'.join([f'    Validation {k}: {v}' for k, v in metrics.items()]))
