import numpy as np
from typing import List

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier:
    def __init__(self, models: List[object], final_model: object):
        """
        Initialize the StackingClassifier with a list of models and a final model.

        :param models: list of initialized models
        :param final_model: initialized final model
        """
        self.models = models
        self.final_model = final_model

    def fit(self, dataset: Dataset, y: np.ndarray) -> 'StackingClassifier':
        """
        Fit each model in the model set and the final model to the dataset.

        :param dataset: features of the training data
        :param y: labels of the training data
        :return: self: StackingClassifier
        """
        # fit the models from the model set
        for model in self.models:
            model.fit(dataset, y)

        # get predictions from each model
        predictions = []
        for model in self.models:
            model_predictions = model.predict(dataset)
            predictions.append(model_predictions)
        predictions = np.array(predictions)

        # fit the final model using the predictions from the model set
        self.final_model.fit(predictions, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from each model in the model set and the final model.

        :param X: features of the data to predict the labels of
        :return: the predicted labels
        """
        # get predictions from each model in the model set
        predictions = []
        for model in self.models:
            model_predictions = model.predict(X)
            predictions.append(model_predictions)
        predictions = np.array(predictions)

        # get the final predictions using the final model and the predictions from the model set
        final_predictions = self.final_model.predict(predictions)
        return final_predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Returns the accuracy of the model.

        :param X: features of the data to evaluate the model on
        :param y: labels of the data to evaluate the model on
        :return: Accuracy of the model.
        """
        y_pred = self.predict(X)
        score = accuracy(y, y_pred)
        return score
