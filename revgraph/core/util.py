import numpy as np


def repeat_to_match_shape(a, shape, axis):
    if shape == ():
        return a, 1
    axis = list(axis) if isinstance(axis, tuple) else axis
    new_shape = np.array(shape)
    new_shape[axis] = 1
    num_reps = np.prod(np.array(shape)[axis])
    return np.broadcast_to(np.reshape(a, new_shape), shape), num_reps
