#!/usr/bin/env python3

import random
from nn_python.util import *
from collections.abc import Iterable


#######################################################
# Validator functions for datastructures
#######################################################
# Validators

def is_signal_in(x):
    """
    Used for validating an input signal

    :param weight: any value
    :returns: boolean indicating if the input is an input signal
    """
    return (isinstance(x, float) or
            isinstance(x, int))


def is_signal_out(x):
    """
    Used for validating an output signal

    :param weight: any value
    :returns: boolean indicating if the input is an output signal
    """
    return ((isinstance(x, float) or
             isinstance(x, int)) and
            x <= 1.0 and
            x >= -1.0)


def is_delta(delta):
    """
    Used for validating a delta

    :param weight: any value
    :returns: boolean indicating if the input is a delta
    """
    return (isinstance(delta, float) or
            isinstance(delta, int))


def is_weight(weight):
    """
    Used for validating a weight

    :param weight: any value
    :returns: boolean indicating if the input is a weight
    """
    return (isinstance(weight, float) or
            isinstance(weight, int))


def is_bias(bias):
    """
    Used for validating a bias

    :param bias: any value
    :returns: boolean indicating if the input is a bias
    """
    return ((isinstance(bias, float) or
             isinstance(bias, int))
            and bias <= 1.0 and bias >= 0.0)


def is_neuron(neuron):
    """
    Used for validating a neuron datastructure

    :param neuron: any value
    :returns: boolean indicating if the input is a neuron
    """
    # sanity check the input
    if not isinstance(neuron, dict):
        return False
    # sanity check, that keys exist
    if 'bias' not in neuron or 'weights' not in neuron:
        return False

    # verify bias
    valid = is_bias(neuron.get('bias', -1))

    # verify weights
    for w in neuron['weights']:
        valid = valid and is_weight(w)
    valid = valid and len(neuron['weights']) > 0

    return valid


def is_layer(layer):
    """
    Used for validating a layer datastructure.
    Layers contain neurons and are contained by by neural networks

    :param layer: any value
    :returns: boolean indicating if the input is a layer
    """
    if not isinstance(layer, Iterable):
        return False
    valid = True
    for neuron in layer:
        valid = valid and is_neuron(neuron)
    return valid


def is_nn(nn):
    """
    Used for validating a neural network datastructure.
    Neural networks contain layers of neurons

    :param nn: any value
    :returns: boolean indicating if the input is a neural network
    """
    # check if every
    # collection of collections of neural networks
    valid = True
    for layer in nn:
        valid = valid and is_layer(layer)
    if valid:
        return nn
    else:
        return False


# Domain functions for primitives

def weight():
    '''
    Create a new randomly generated weight.
    Note: the range is inclusive at both ends

    :returns: A float between [-1.0 and 1.0]
    '''
    return random.uniform(-1.0, 1.0)


def bias():
    '''
    Create a new randomly generated bias.
    Note: the range is inclusive at both ends

    :returns: A float between [-0.0 and 1.0]
    '''
    return random.uniform(0.0, 1.0)


# Domain functions for data structures

def logistic_sigmoid(x):
    """
    Implementation of logistic sigmoid function

    :param x: Any finite number
    :returns: A value constrained between (0.0 and 1.0)
    """
    return 1.0 / (1.0 + math.e ** -x)


def logistic_sigmoid_prime(x):
    """
    Implementation of the derivative of the logistic sigmoid function

    :param x: Any finite number
    :returns: A value constrained between (0.0 and 1.0)
    """
    return x * (1.0 - x)


# Data structure creation functions
def neuron(num_weights):
    """
    Create and return a new neuron

    :param num_weights: The number of weights to create
    :returns: A new neuron containing a number of randomly generated weights
    """
    return {'bias': bias(),
            'weights': tuple(repeatedly(num_weights, weight))}


def nn_old(num_inputs, num_outputs,
           num_hidden_layers=1,
           fire=logistic_sigmoid,
           fire_prime=logistic_sigmoid_prime):
    """
    Create and return a new neural network

    :param num_inputs: The number of inputs the neural network will take.
    :param num_outputs: The number of outputs the neural network will generate.
    :param num_hidden_layers: The number of hidden layers to generate.
    :param fire: The firing function used for each neuron.
      (default logistic_sigmoid)
    :param fire_prime: The derivative of the firing function for each neuron
      (default logistic_sigmoid_prime)
    :returns: A new neural network
    """
    prev_layer_neurons = num_inputs
    hidden_layers = []
    # Generate the number of hidden layers
    for i in range(num_hidden_layers):
        # create each neuron in the hidden layer
        hidden_neurons_in_layer = random.randint(min(num_inputs, num_outputs),
                                                 max(num_inputs, num_outputs))
        hidden_layer = [neuron(prev_layer_neurons)
                        for _ in range(hidden_neurons_in_layer)]

        # add the new hidden layer, and prepare for the next layer
        hidden_layers.append(hidden_layer)
        prev_layer_neurons = hidden_neurons_in_layer

    # create output layer of neurons
    output_layer = [neuron(prev_layer_neurons) for _ in range(num_outputs)]
    _nn = hidden_layers
    _nn.append(output_layer)

    return {'fire': fire,
            'fire_prime': fire_prime,
            'neurons': _nn}


def layer(num_neurons, prev_layer_neurons):
    """
    Create a new layer

    :param num_neurons: The number of neurons in the layer
    :param prev_layer_neurons: The number of neurons in the previous layer:
    :returns: A tuple of neurons representing the new layer
    """
    _layer = []
    for _ in range(num_neurons):
        _layer.append(neuron(prev_layer_neurons))
    return tuple(_layer)


def nn(neurons,
       fire=logistic_sigmoid,
       fire_prime=logistic_sigmoid_prime):
    """
    Create and return a new neural network

    :param num_inputs: The number of inputs the neural network will take.
    :param num_outputs: The number of outputs the neural network will generate.
    :param num_hidden_layers: The number of hidden layers to generate.
    :param fire: The firing function used for each neuron.
      (default logistic_sigmoid)
    :param fire_prime: The derivative of the firing function for each neuron
      (default logistic_sigmoid_prime)
    :returns: A tuple of layers representing the new neural network
    """

    num_inputs, *xs = neurons

    _nn = []
    prev_num_neurons = num_inputs

    # generate hidden layers (if any) and output layer
    for x in xs:
        _layer = layer(x, prev_num_neurons)
        _nn.append(_layer)
        prev_num_neurons = x

    return tuple(_nn)
