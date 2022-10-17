import numpy as np
from typing import Union
from src.si.data.dataset import Dataset

class PCA:
    """
    A class that performs Principal Component Analysis (PCA) on a given dataset using the Singular Value Decomposition (SVD) method.
    """
    def __init__(self, n_components: int) -> None:
        """
        Initializes the PCA object.
        :param n_components: Number of components to be considered and returned from the analysis.
        """
        self.n_components = n_components

        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset: Dataset) -> 'PCA':
        """
        Fits the data and stores the mean values of each sample, the principal components, and the explained variance.
        :param dataset: Dataset object containing the data.
        :return: The PCA object.
        """
        self.mean = np.mean(dataset.X, axis=0)
        self.centered = dataset.X - self.mean

        U, S, Vt = np.linalg.svd(self.centered, full_matrices=False)

        self.components = Vt[:, :self.n_components]

        EV = (S ** 2) / (len(dataset.X) - 1)
        self.explained_variance = EV[:self.n_components]

        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Returns the calculated reduced dataset.
        :param dataset: Dataset object containing the data.
        :return: A Dataset object containing the reduced data.
        """
        self.mean = np.mean(dataset.X, axis=0)
        centered = dataset.X - self.mean

        X_reduced = np.dot(centered, self.components.T)

        return Dataset(X_reduced, dataset.y, dataset.features, dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
            """
            Fits the data and transforms it.
            :param dataset: Dataset object containing the data.
            :return: A Dataset object containing the reduced data.
            """

            self.fit(dataset)
            return self.transform(dataset)