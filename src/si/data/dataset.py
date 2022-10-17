import numpy as np
import pandas as pd
from typing import Tuple, Sequence


class Dataset:

    def __init__(self, X, y, features, label):
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
        return self.X.shape

    def has_label(self):
        if self.y is not None:
            return True
        return False

    def get_classes(self):
        if self.y is None:
            return
        return np.unique(self.y)

    def get_mean(self):
        return np.mean(self.X, axis=0)

    def get_variance(self):
        return np.var(self.X, axis=0)

    def get_median(self):
        return np.median(self.X, axis=0)

    def get_min(self):
        return np.min(self.X, axis=0)

    def get_max(self):
        return np.max(self.X, axis=0)


    def summary(self):
        return pd.DataFrame(
            {'mean': self.get_mean(),
             'variance': self.get_variance(),
             'median': self.get_median(),
             'min': self.get_min(),
             'max': self.get_max()}
        )


    # Exercicio 2
    def dropna(self):
        mask = np.logical_not(np.any(np.isnan(self.x), axis=1))
        self.X = self.X[mask, :]
        return self.X

    def fillna(self, value):
        self.x = np.nan_to_num(self.X, value)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame
        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name
        Returns
        -------
        Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame
        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data
        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name
        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)

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
