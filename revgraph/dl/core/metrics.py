from .utils import *


@register
def categorical_accuracy() -> Metric:
    def function(y_true: rc.tensor,
                 y_pred: rc.tensor) -> rc.tensor:
        return rc.sum(rc.argmax(y_true, axis=1) == rc.argmax(y_pred, axis=1)) / rc.len(y_true)
    return function
