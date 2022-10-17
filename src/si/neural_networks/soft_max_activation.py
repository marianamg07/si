import numpy as np


class SoftMaxActivation:
    def __init__(self):
        """
        Initializes the SoftMaxActivation class.
        """
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the probability of occurrence of each class using the following formula:
        e^zi/(σ), where zi = X - max(X) and σ = sum of the exponential of the zi vector.

        Parameters:
        - X (np.ndarray): Input matrix of shape (batch_size, num_classes).

        Returns:
        - np.ndarray: Probability matrix of shape (batch_size, num_classes).
        """
        # Subtract the maximum value of X from all elements of X to ensure numerical stability
        X_max = np.max(X, axis=1, keepdims=True)
        X_stable = X - X_max

        # Calculate the exponential of X_stable
        exp_X_stable = np.exp(X_stable)

        # Calculate the sum of the exponential of X_stable
        sum_exp_X_stable = np.sum(exp_X_stable, axis=1, keepdims=True)

        # Calculate the probability of occurrence of each class
        probabilities = exp_X_stable / sum_exp_X_stable

        return probabilities
