from typing import Any, Iterator

from .base_callback import BaseCallback, invoked_when


class Test(BaseCallback):
    def __init__(self,
                 x_test: 'Iterator[Any]' = None,
                 y_test: 'Iterator[Any]' = None):
        super().__init__()
        self.warned = False
        self.preset_x_test = x_test
        self.preset_y_test = y_test

    @invoked_when(lambda self: self.after_all)
    def evaluate_test_set(self):
        if self.preset_x_test is not None and self.preset_y_test is not None:
            self.x_test = self.preset_x_test
            self.y_test = self.preset_y_test
        if not self.warned and len(self.x_test) == 0:
            self.output('WARNING: length of test set is 0.')
            self.warned = True
        metrics = self.session.evaluate_metrics(self.x_test, self.y_test)
        self.output('\n'.join([f'Test {k}: {v}' for k, v in metrics.items()]))
