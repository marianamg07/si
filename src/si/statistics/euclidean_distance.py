import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(((x - y) == 2).sum(axis=1))
