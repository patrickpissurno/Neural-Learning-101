# Source-code obtained from the first example in http://iamtrask.github.io/2015/07/12/basic-python-network/
# Keep in mind that this code was written in Python 2
import numpy as np

# sigmoid function (https://en.wikipedia.org/wiki/Sigmoid_function)
def nonlin(x, deriv = False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# input dataset
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# output dataset
y = np.array([
    [0, 0, 1, 1]
    ]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0 (between -1 and 1)
syn0 = 2 * np.random.random((3, 1)) - 1

for i in xrange(10000):

    # forward propagation
    layer0 = X
    layer1 = nonlin(np.dot(layer0, syn0))

    # how much did we miss?
    layer1_err = y - layer1

    # multiply how much we missed by the
    # slope (derivative) of the sigmoid at the values in layer1
    layer1_delta = layer1_err * nonlin(layer1, True)

    # update weights
    syn0 += np.dot(layer0.T, layer1_delta)

print layer1

# -------------------------------------------------------------------------------
#                General information about this piece of code
# -------------------------------------------------------------------------------
#
# X => Input dataset matrix where each row is a training example
# y => Output dataset matrix where each row is a training example
#
# layer0 => First Layer of the Network, specified by the input data
# layer1 => Second Layer of the Network, otherwise known as the hidden layer
#
# syn0 => First layer of weights, Synapse 0, connecting layer0 to layer1
#
# * => Elementwise multiplication, so two vectors of equal size are multiplying
#      corresponding values 1-to-1 to generate a final vector of identical size
#
# - => Elementwise subtraction, so two vectors of equal size are subtracting 
#      corresponding values 1-to-1 to generate a final vector of identical size
#
# x.dot(y) => If x and y are vectors, this is a dot product. If both are matrices,
#             it's a matrix-matrix multiplication. If only one is a matrix, then
#             it's vector matrix multiplication
#
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
#                                     FAQ
# -------------------------------------------------------------------------------
#
# What is a sigmoid function? A function that maps any value to a value between 0
# and 1. We use it to convert numbers to probabilities
#
# When you see 'matrix.T', the '.T' stands for 'transpose' and returns the
# transposed version of that matrix
#
# Why do you call 'np.random.seed(1)'? To get numbers randomly distributed in
# the exactly same way each time you train
#
# What is 'syn0'? This is our weight matrix for this neural network. It's called
# 'syn0' to imply 'synapse zero'. Since we only have 2 layers (input and output),
# we only need one matrix of weights to connect them. Its dimension is (3,1) because
# we have 3 inputs and 1 output
#
# For more information please check out the original article:
# http://iamtrask.github.io/2015/07/12/basic-python-network/
# -------------------------------------------------------------------------------
