import numpy as np
from src.helper_functions import relu, sigmoid, relu_derivative, sigmoid_derivative

class Neuron:

    def __init__(self, nin: int, activation: str = 'relu'):
        self.activation = activation

        if activation == 'relu':
            # He initialization for ReLU
            limit = np.sqrt(2.0 / nin)
        else:
            # Xavier initialization for sigmoid/tanh
            limit = np.sqrt(6.0 / (nin + 1))

        self.weights = np.random.uniform(-limit, limit, nin).astype(np.float64)
        self.bias = np.random.normal()

        self.grads = np.zeros(nin, dtype=np.float64)
        self.gradb = np.float64(0.0)
        self.inputs = np.array([], dtype=np.float64)

        self.activ = 0
        self.delta = 0

    def __call__(self, x):
        interm = np.dot(self.weights, x) + self.bias
        self.inputs = np.array(x, dtype=np.float64)
        self.res = interm

        if self.activation == 'relu':
            res = relu(interm)
        else:  # sigmoid
            res = sigmoid(interm)

        self.activ = res
        return res

    def activation_derivative(self):
        if self.activation == 'relu':
            return relu_derivative(self.res)
        else:
            return sigmoid_derivative(self.activ)

    def __repr__(self):
        return f'Neuron(Weights:{self.weights}, Bias:{self.bias})'