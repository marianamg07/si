import numpy as np
import pandas as pd


class Dataset:

    def __init__(self, X, y, features, label):
        self.x = X
        self.Y = y
        self.features = features
        self.label = label

    def shape(self):
        return self.x.shape

    def has_label(self):
        if self.Y is not None:
            return True
        return False

    def get_classes(self):
        if self.Y is None:
            return
        return np.unique(self.Y)

    def get_mean(self):
        return np.mean(self.x, axis=0)

    def get_variance(self):
        return np.var(self.x, axis=0)

    def get_median(self):
        return np.median(self.x, axis=0)

    def get_min(self):
        return np.min(self.x, axis=0)

    def get_max(self):
        return np.max(self.x, axis=0)
    def summary(self):
        return pd.DataFrame(
            {'mean': self.get_mean(),
            'variance': self.get_variance(),
            'median': self.get_median(),
            'mean': self.get_mean(),
            'min': self.get_min(),
            'max': self.get_max()}
        )

if __name__ == '__main__':
    x = np.array([[1, 2, 3], [1, 2, 3]])
    Y = np.array([1, 2])
    features = ['A', 'B', 'C']
    label = 'y'
    dataset = Dataset(X=x, y=Y, features=features, label=label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.summary())

