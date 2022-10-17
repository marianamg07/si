from typing import Callable, Union

import numpy as np

from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor:
    """
    KNN Classifier
    The k-Nearst Neighbors classifier is a machine learning model that classifies new samples based on
    a similarity measure (e.g., distance functions). This algorithm predicts the classes of new samples by
    looking at the classes of the k-nearest samples in the training data.
    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use
    Attributes
    ----------
    dataset: np.ndarray
        The training data
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Initialize the KNN classifier
        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        # parameters
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        Returns
        -------
        self: KNNClassifier
            The fitted model
        """
        self.dataset = dataset
        return self

    def predict_1(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the closest label of the given sample
        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label of
        Returns
        -------
        label: str or int
            The closest label
        """
        # compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)

        # gets the indexes from k with the nearest neighbors
        k_n_n = np.argsort(distances)[:self.k]

        # gets the values in y using the indexes from k nearest neighbors
        k_n_n_labels = self.dataset.y[k_n_n]

        # gets the mean from the values in y (labels)
        mvy = np.average(k_n_n_labels)
        return mvy

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of
        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """
        return np.apply_along_axis(self.predict_1, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        It returns the rmse of the model on the given dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on
        Returns
        -------
        accuracy: float
            The rmse of the model
        """
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)



