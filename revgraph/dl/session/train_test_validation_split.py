from revgraph.dl.core.utils import *


def train_test_validation_split(x: np.array,
                                y: np.array,
                                train: float,
                                test: float,
                                validation: float) -> List[Tuple[np.array, np.array]]:
    validate((
        train + test + validation == 1.0,
        'Sum of train, test and validation must be 1.0'
    ), (
        len(x) == len(y),
        f'\'x\' and \'y\' must have the same length, instead of {len(x)} and {len(y)}'
    ))
    n = len(x)
    test_starts = int(train * n) + 1
    x_train, y_train = x[:test_starts], y[:test_starts]
    validation_starts = int((train + test) * n) + 1
    x_test, y_test = x[test_starts:validation_starts], y[test_starts:validation_starts]
    x_valid, y_valid = x[validation_starts:], y[validation_starts:]
    return [(x_train, y_train),
            (x_test, y_test),
            (x_valid, y_valid)]
