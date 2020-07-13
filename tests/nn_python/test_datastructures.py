#!/usr/bin/python3
from nn_python.datastructures import *


#######################################################
# Test datastructures
#######################################################
def test_signals():
    # test validation function
    assert(is_signal_in(0) and
           is_signal_in(1) and
           is_signal_in(1.0) and
           is_signal_in(-50000.0) and
           is_signal_in(12345678))
    assert(is_signal_out(0) and
           is_signal_out(1) and
           is_signal_out(1.0) and
           is_signal_out(-1.0) and
           not is_signal_out(-50000.0) and
           not is_signal_out(12345678))


def test_delta():
    # test validation function
    assert(is_delta(0) and
           is_delta(1) and
           is_delta(1.0) and
           is_delta(-50000.0) and
           is_delta(12345678))


def test_weight():
    # test validation function
    assert(is_weight(0) and
           is_weight(1) and
           is_weight(1.0) and
           is_weight(-50000.0) and
           is_weight(12345678))


def test_bias():
    # test validation function
    assert(is_bias(0) and
           is_bias(1) and
           is_bias(1.0) and
           not is_bias(-50000.0) and
           not is_bias(12345678))


def test_neuron():
    simple_neuron = {"weights": [1], "bias": 1}
    bad_no_weights = {"bias": 1}
    bad_empty_weights = {"weights": [], "bias": 1}
    bad_no_bias = {"weights": []}
    # test validation function
    assert(is_neuron(simple_neuron) and
           not is_neuron(bad_no_weights) and
           not is_neuron(bad_empty_weights) and
           not is_neuron(bad_no_bias))
    # test generator function
    assert(is_neuron(neuron(3)))


def test_layer():
    simple_neuron = {"weights": [1], "bias": 1}
    simple_layer = (simple_neuron,)
    bad_layer_invalid_neuron = (simple_neuron, False, simple_neuron)
    # test validation function
    assert(is_layer(simple_layer) and
           not is_layer(bad_layer_invalid_neuron))
    # test generator function
    assert(is_layer(layer(3, 2)))


def test_nn():
    simple_neuron = {"weights": [1], "bias": 1}
    simple_layer = (simple_neuron,)
    simple_nn = (simple_layer, simple_layer, simple_layer)
    bad_nn_invalid_layer = (simple_layer, False, simple_layer)
    # test validation function
    assert(is_nn(simple_nn) and
           not is_nn(bad_nn_invalid_layer))

    # test generator function

    # nn with multiple hidden layers
    assert(is_nn(nn([3, 7, 3, 2])))
    # nn with one hidden layer
    assert(is_nn(nn([2, 1, 4])))
    # nn with no hidden layers
    assert(is_nn(nn([5, 1])))
    # nn with only input layers
    assert(not is_nn(nn([5, ])))
