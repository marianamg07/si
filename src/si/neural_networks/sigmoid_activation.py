import numpy as np

class SigmoidActivation:
    def __init__(self):
        pass

    def forward(self, input_data):
        return 1 / (1 + np.exp(-input_data))

    def backward(self, input_data, grad_output):
        output = self.forward(input_data)
        return grad_output * output * (1 - output)
