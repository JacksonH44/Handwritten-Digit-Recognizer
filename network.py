'''
A module that implements the stochastic gradient descent learning algorithm for a feedforward neural network. Gradients are calculated using backpropagation. 

Reference: 'Neural Networks and Deep Learning' by Michael Nielson
'''

import numpy as np

# Sigmoid Function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Network(object):

    # Constructor
    # Sizes: number of neurons in respective layers (i.e. [2, 3, 1] is 2 input, 3 hidden, 1 output)
    # biases: column vector for biases for all layers but input
    # weights: y by x matrices such that w_jk is the connection from the kth neuron in the ith layer to the jth neuron in the i - 1th layer
    def __init__(self, sizes) -> None:
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # Return the output of the network for an input a
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a_p = sigmoid(np.dot(w, a) + b)
        return a_p