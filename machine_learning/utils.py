import numpy as np


def find_nearest(array, value):
    array = np.asarray(array)
    diffs = (np.abs(array - value))
    return np.unravel_index(np.argmin(diffs), diffs.shape)

