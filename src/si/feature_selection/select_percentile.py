from typing import Callable, Tuple, Union
import numpy as np
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile:
    def __init__(self, percentile: float = 0.25, score_func: Callable[
        [np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = f_classification) -> None:
        """
        Parameters:
        - percentile: percentile for features to select (float, default 0.25)
        - score_func: variance analysis function (Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]], default f_classification)

        Estimated parameters:
        - F: the value of F for each feature estimated by score_func (np.ndarray)
        - p: the p-value for each feature estimated by score_func (np.ndarray)
        """
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Estimates the F and p for each feature using the scoring_func.

        Parameters:
        - dataset: a given dataset (Dataset)

        Returns:
        self
        """
        self.F, self.p = self.score_func(dataset.X, dataset.y)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Selects the features with the highest F value up to the indicated percentile.
        (for a dataset with 10 features and a 50% percentile, the transform should select
        the 5 features with higher F value)

        Parameters:
        - dataset: a given dataset (Dataset)

        Returns:
        dataset
        """
        num_features = int(self.percentile * dataset.X.shape[1])
        top_features = np.argpartition(-self.F, num_features)[:num_features]
        return Dataset(X=dataset.X[:, top_features], y=dataset.y, features=[dataset.features[i] for i in top_features],
                       label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Runs the fit and then the transform.

        Parameters:
        - dataset: a given dataset (Dataset)

        Returns:
        transformed dataset
        """
        self.fit(dataset)
        return self.transform(dataset)
