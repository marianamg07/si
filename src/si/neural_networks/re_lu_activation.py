import numpy as np


class ReLUActivation:
    def __init__(self):
        """
        Initializes the ReLUActivation class.
        """
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the rectified linear ratio of the input matrix X. That is, the positive
        part of each element in X is taken.

        Parameters:
        - X (np.ndarray): Input matrix of shape (batch_size, num_features).

        Returns:
        - np.ndarray: Output matrix of shape (batch_size, num_features).
        """
        # Calculate the rectified linear ratio of X
        self.output = np.maximum(0, X)

        return self.output

    def backward(self, error: np.ndarray) -> np.ndarray:
        """
        Propagates the error through the ReLU activation layer.

        Parameters:
        - error (np.ndarray): Error matrix of shape (batch_size, num_features).

        Returns:
        - np.ndarray: Propagated error matrix of shape (batch_size, num_features).
        """
        # Replace error values greater than 0 with 1
        error[error > 0] = 1

        # Replace error values less than 0 with 0
        error[error < 0] = 0

        # Element-wise multiplication between the error and the previous output
        propagated_error = error * self.output

        return propagated_error
