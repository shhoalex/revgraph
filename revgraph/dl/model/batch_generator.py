from typing import *

import numpy as np


def batch_generator(x: np.array,
                    y: np.array,
                    batch_size: Optional[int] = None,
                    shuffle: bool = False
                    ) -> Generator[Tuple[np.array, np.array], None, None]:
    if shuffle:
        perm = np.random.permutation(len(x))
        x = x[perm]
        y = y[perm]

    if batch_size is None:
        yield (x, y)
    else:
        for i in range(0, len(x), batch_size):
            batch = slice(i, i+batch_size)
            if i+batch_size <= len(x):
                yield (x[batch], y[batch])
