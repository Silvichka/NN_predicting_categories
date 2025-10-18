from src.Neuron import Neuron
import numpy as np

class Layer:

    def __init__(self, nin: int, nout: int, activation: str = 'relu'):
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]
        self.activation = activation

    def __call__(self, x):
        res = np.array([n(x) for n in self.neurons])
        return res if len(res) == 2 else res

    def __repr__(self):
        return f'Layer: {[x for x in self.neurons]}'

    def parameters(self):
        return [x for x in self.neurons]