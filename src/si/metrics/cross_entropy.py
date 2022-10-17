import numpy as np


def cross_entropy(y_true, y_pred):
    """
    Calculate the cross entropy error measure.

    Parameters:
    - y_true (numpy array): The real values.
    - y_pred (numpy array): The predicted values.

    Returns:
    - float: The cross entropy error measure.
    """
    cross_entropy = - np.sum(y_true) * np.log(y_pred) / len(y_true)
    return cross_entropy


def cross_entropy_derivative(y_true, y_pred):
    """
    Calculate the derivative of the cross entropy error measure.

    Parameters:
    - y_true (numpy array): The real values.
    - y_pred (numpy array): The predicted values.

    Returns:
    - numpy array: The derivative of the cross entropy error measure.
    """
    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred)) / len(y_true)
